"""Geração por tarefa: sem marca, STONE, híbrido (STONE + H > tau) e negativo humano.

    python src/gerar.py --config configs/smoke.yaml
    python src/gerar.py --config configs/apps.yaml --inicio 0 --fim 200      # sorteio 600 vem do YAML
    python src/gerar.py --config configs/humaneval.yaml --n_amostras 5
    python src/gerar.py --config configs/apps.yaml --listar                  # só mostra a seleção

Cada (tarefa, braço, amostra) vira resultados/<rodada>/<conjunto>/<divisão>/<chave>_<braço>_s<i>.json.gz
(o humano: <chave>_humano.json.gz), gravado de forma atômica. Rodar de novo retoma a rodada;
acrescentar braços, deltas, taus ou amostras reaproveita a mesma pasta.
"""
import argparse
import functools
import gc
import importlib.metadata as im
import json
import logging
import math
import os
import platform
import sys
import time
from datetime import datetime
from itertools import zip_longest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig,
                          LogitsProcessor, LogitsProcessorList, StoppingCriteria,
                          StoppingCriteriaList)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comum  # noqa: E402
from comum import (CONJUNTOS, FONTES, PARADA, corte, digest, fetch, fronteira,  # noqa: E402
                   read_gz, write_gz, write_text_atomic)

PROTOCOLO = 'scripts-2'   # notebook v5: 'v5.1'; scripts-1: sem híbrido/humano e APPS no formato antigo
SYN = set(('True False None and as assert async await break class continue def del elif else '
           'except finally for from global if import in is lambda nonlocal not or pass raise return '
           'try while with yield int float complex str bytes bool list tuple set dict NoneType').split())
PUNCT = set('()[]{},:.;@->=+*/%<>!&|^~')
log = logging.getLogger('gerar')


# ------------------------------------------------------------------ inventário
def carregar_conjunto(nome, tok, model_id, revision):
    """Inventário de um conjunto, em cache em dados/ (depende do tokenizador)."""
    import gzip
    index = comum.DADOS / f'inventario_{nome}_{comum.chave_inventario(nome, model_id, revision)}.json.gz'
    if index.exists():
        return read_gz(index)
    ntok = lambda s: len(tok.encode(s, add_special_tokens=False, verbose=False))  # noqa: E731
    tasks = []
    if nome == 'humaneval':
        with gzip.open(fetch('HumanEval+', log=log.info), 'rt', encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                tasks.append(dict(dataset='HumanEval+', split='test', id=d['task_id'], prompt=d['prompt'],
                                  formato='continuacao', referencia=d['canonical_solution'],
                                  ref_tokens=ntok(d['canonical_solution']), difficulty='',
                                  target=d['entry_point']))
        assert len(tasks) == 164, f'HumanEval+: esperados 164, encontrados {len(tasks)}'
    elif nome == 'apps':
        for split in ('test', 'train'):
            n = 0
            with fetch(f'APPS/{split}', log=log.info).open(encoding='utf-8') as f:
                for line in f:
                    n += 1
                    t = comum.apps_tarefa(json.loads(line), split)
                    if t is None:
                        continue
                    codes = t.pop('solucoes')
                    lens = []
                    for i in range(0, len(codes), 32):
                        batch = tok(codes[i:i + 32], add_special_tokens=False, truncation=False, verbose=False)
                        lens.extend(map(len, batch.input_ids))
                    t['ref_tokens'] = float(np.median(lens)) if lens else None
                    tasks.append(t)
            if n != 5000:
                raise ValueError(f'APPS {split}: esperados 5000 registros, encontrados {n}.')
    elif nome == 'classeval':
        cols = ['task_id', 'skeleton', 'solution_code', 'class_name', 'test']
        for d in pd.read_parquet(fetch('ClassEval', log=log.info), columns=cols).to_dict('records'):
            prompt = ('Implement the complete Python class described by the following skeleton. '
                      'Return the entire class as Python code.\n\n' + d['skeleton'] + '\n\n# Implementation\n')
            tasks.append(dict(dataset='ClassEval', split='test', id=d['task_id'], prompt=prompt,
                              formato='bloco', referencia=d['solution_code'], testes=d['test'],
                              ref_tokens=ntok(d['solution_code']), difficulty='', target=d['class_name']))
        assert len(tasks) == 100, f'ClassEval: esperadas 100, encontradas {len(tasks)}'
    write_gz(index, tasks)
    return tasks


def inventario(cfg, tok, revision):
    tasks = []
    for nome in cfg['datasets']:
        tasks.extend(carregar_conjunto(nome, tok, cfg['model_id'], revision))
    grupos = {}
    for t in tasks:
        t['key'] = digest([t['dataset'], t['split'], t['id']])[:24]   # mesma chave do notebook
        grupos.setdefault((t['dataset'], t['split']), []).append(t)
    assert len({t['key'] for t in tasks}) == len(tasks), 'IDs duplicados'
    inv = pd.DataFrame(tasks).drop(columns=['prompt', 'referencia', 'testes'], errors='ignore')
    inv['tem_referencia'] = [bool(t.get('referencia')) for t in tasks]
    inv['faixa_ref'] = ''
    for _, frame in inv.groupby(['dataset', 'split'], sort=False):
        q1, q2 = frame.ref_tokens.quantile([1 / 3, 2 / 3])
        x = frame.ref_tokens
        inv.loc[frame.index, 'faixa_ref'] = np.select(
            [x.isna(), x <= q1, x <= q2], ['sem referência', 'curta', 'média'], default='longa')
    return tasks, grupos, inv


# --------------------------------------------------------- marca e parada
class TraceSTONE(LogitsProcessor):
    """Processador do notebook v5, generalizado para os três braços.

    Em toda posição: tokens FIM recebem -inf; H vem desses logits (antes da marca e dos
    filtros de amostragem); um candidato top-k é sorteado com gerador próprio e decide a
    elegibilidade do STONE (candidato não sintático). O híbrido exige também H > tau.
    - sem_marca: só registra;  stone: +delta nos verdes se elegível pelo STONE;
    - hibrido: +delta nos verdes se elegível pelo STONE e H > tau.
    """

    def __init__(self, ctx, tipo, delta, tau, seed, device):
        self.ctx, self.tipo, self.delta, self.tau = ctx, tipo, delta, tau
        self.probe = torch.Generator(device=device).manual_seed(seed)
        self.entropies, self.candidatos, self.eligible = [], [], []

    def __call__(self, input_ids, scores):
        c = self.ctx
        if scores.shape[-1] != c.vocab_verde:
            raise RuntimeError(f'Logits com {scores.shape[-1]} posições, lista verde com {c.vocab_verde}.')
        if c.suprimir is not None:
            scores = scores.clone()
            scores[:, c.suprimir] = -math.inf
        values = scores[0].float()
        logp = torch.log_softmax(values, dim=-1)
        p = logp.exp()
        h = -torch.where(p > 0, p * logp, 0).sum()
        self.entropies.append(h)
        shortlist, ids = torch.topk(values / c.cfg['temperature'], min(c.cfg['top_k'], values.numel()))
        sample = torch.multinomial(shortlist.softmax(-1), 1, generator=self.probe)
        cand = int(ids[sample].item())
        eligible = not c.syntactic(cand)
        self.candidatos.append(cand)
        self.eligible.append(eligible)
        marcar = (self.tipo == 'stone' and eligible) or (
            self.tipo == 'hibrido' and eligible and float(h) > self.tau)
        if marcar and self.delta:
            prev = int(input_ids[0, -1].item())
            scores = scores.clone()
            scores[0] += self.delta * c.mask_cpu(prev).to(scores.device, dtype=scores.dtype)
        return scores


class ParadaFronteira(StoppingCriteria):
    """Para quando a função (HumanEval+, APPS) ou a classe (ClassEval) termina.
    A verificação decodifica a resposta inteira, por isso roda a cada N tokens."""

    def __init__(self, ctx, task, plen, a_cada):
        self.ctx, self.task, self.plen, self.a_cada = ctx, task, plen, a_cada
        self.disparou = False

    def __call__(self, input_ids, scores, **kwargs):
        n = input_ids.shape[1] - self.plen
        if not self.disparou and n > 0 and n % self.a_cada == 0:
            ids = input_ids[0, self.plen:].tolist()
            k = next((i for i, t in enumerate(ids) if t in self.ctx.especiais), len(ids))
            self.disparou = fronteira(self.task, self.ctx.decode(ids[:k])) is not None
        return torch.full((input_ids.shape[0],), self.disparou, dtype=torch.bool, device=input_ids.device)


class Contexto:
    """Modelo, tokenizador e funções que dependem deles."""

    def __init__(self, cfg, tok, model, vocab_verde, suprimir_ids, eos):
        self.cfg, self.tok, self.model = cfg, tok, model
        self.vocab_verde = vocab_verde
        self.suprimir = torch.tensor(suprimir_ids, device=model.device) if suprimir_ids else None
        self.suprimir_set = set(suprimir_ids)
        self.eos = eos
        self.especiais = set(tok.all_special_ids) | set(eos)
        self.context = int(model.config.max_position_embeddings)
        self.syntactic = functools.lru_cache(maxsize=65536)(self._syntactic)
        self.mask_cpu = functools.lru_cache(maxsize=128)(self._mask_cpu)

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def _syntactic(self, tid):
        s = self.tok.decode([tid], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        return not s or s in SYN or all(ch in PUNCT for ch in s)

    def _mask_cpu(self, prev):
        vocab = self.vocab_verde
        rng = torch.Generator(device='cpu').manual_seed(self.cfg['hash_key'] * (prev % vocab))
        green = torch.zeros(vocab, dtype=torch.bool)
        green[torch.randperm(vocab, generator=rng)[:int(vocab * self.cfg['gamma'])]] = True
        return green


def anotar(ctx, ultimo_prompt, ids, H):
    """Campos por posição usados pelo detector, iguais para gerações e humano."""
    counted, green, special = [], [], []
    last = ultimo_prompt
    for tid in ids:
        sp = tid in ctx.especiais
        special.append(sp)
        counted.append(not sp and not ctx.syntactic(tid))
        green.append(bool(ctx.mask_cpu(last)[tid]) if not sp else False)
        last = tid
    return dict(H=H, counted=counted, green=green, special=special)


def seed_amostra(cfg, task, amostra):
    return int(digest([cfg['seed'], task['key'], amostra])[:8], 16)   # igual em todos os braços


@torch.inference_mode()
def generate(ctx, task, braco, amostra, max_new):
    cfg, tok, model = ctx.cfg, ctx.tok, ctx.model
    enc = tok(task['prompt'], return_tensors='pt', add_special_tokens=False,
              return_token_type_ids=False, truncation=False)
    plen = enc.input_ids.shape[1]
    if not 0 < plen < ctx.context:
        return dict(status='prompt_sem_espaco_no_contexto', prompt_tokens=plen)
    max_efetivo = min(max_new, ctx.context - plen)
    seed = seed_amostra(cfg, task, amostra)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    enc = enc.to(model.device)
    proc = TraceSTONE(ctx, braco['tipo'], braco['delta'], braco['tau'], seed ^ 0xA19D, model.device)
    parada = ParadaFronteira(ctx, task, plen, cfg['stop_check_every'])
    pad = tok.pad_token_id if tok.pad_token_id is not None else ctx.eos[0]
    gen = GenerationConfig(max_new_tokens=max_efetivo, do_sample=True, temperature=cfg['temperature'],
                           top_k=cfg['top_k'], top_p=1.0, eos_token_id=ctx.eos, pad_token_id=pad,
                           use_cache=True, output_scores=False, forced_eos_token_id=None)
    t0 = time.perf_counter()
    seq = model.generate(**enc, generation_config=gen, logits_processor=LogitsProcessorList([proc]),
                         stopping_criteria=StoppingCriteriaList([parada]), use_model_defaults=False)
    segundos = time.perf_counter() - t0
    ids = seq[0, plen:].tolist()
    assert len(ids) == len(proc.entropies) == len(proc.eligible)
    vazados = sorted(ctx.suprimir_set.intersection(ids))
    if vazados:
        raise RuntimeError(f'Tokens suprimidos apareceram na saída: {vazados}')
    h = torch.stack(proc.entropies).cpu().tolist() if ids else []
    if ids and ids[-1] in ctx.eos:
        stop = 'eos'
    elif parada.disparou:
        stop = PARADA[task['dataset']]
    elif len(ids) >= max_efetivo:
        stop = 'max_new_tokens' if max_efetivo == max_new else 'contexto'
    else:
        stop = 'outro'
    return dict(status='ok', braco=braco['nome'], tipo=braco['tipo'], delta=braco['delta'], tau=braco['tau'],
                amostra=amostra, seed=seed, prompt_tokens=plen, ids=ids, text=ctx.decode(ids),
                **anotar(ctx, int(enc.input_ids[0, -1].item()), ids, h),
                candidato=proc.candidatos, eligible_stone=proc.eligible,
                eligible_hibrido={str(t): [e and x > t for e, x in zip(proc.eligible, h)] for t in cfg['taus']},
                stop=stop, max_new_tokens=max_efetivo, corte=corte(task, ids, ctx.decode, ctx.especiais),
                segundos=round(segundos, 3))


@torch.inference_mode()
def pontuar_humano(ctx, task, bloco=256):
    """Negativo humano: teacher forcing da referência com o mesmo prompt.
    Prompt e referência são tokenizados separadamente, como prompt e geração."""
    tok, model = ctx.tok, ctx.model
    ref = task.get('referencia')
    if not ref:
        return dict(status='sem_referencia', braco='humano', tipo='humano')
    p_ids = tok(task['prompt'], add_special_tokens=False).input_ids
    r_ids = tok(ref, add_special_tokens=False).input_ids
    if not p_ids or len(p_ids) + len(r_ids) > ctx.context:
        return dict(status='prompt_sem_espaco_no_contexto', braco='humano', tipo='humano',
                    prompt_tokens=len(p_ids))
    t0 = time.perf_counter()
    entrada = torch.tensor([p_ids + r_ids], device=model.device)
    logits = model(input_ids=entrada, logits_to_keep=len(r_ids) + 1).logits[0, :-1]
    if logits.shape[-1] != ctx.vocab_verde:
        raise RuntimeError(f'Logits com {logits.shape[-1]} posições, lista verde com {ctx.vocab_verde}.')
    H = []
    for i in range(0, logits.shape[0], bloco):
        x = logits[i:i + bloco].float()
        if ctx.suprimir is not None:
            x[:, ctx.suprimir] = -math.inf
        logp = torch.log_softmax(x, dim=-1)
        p = logp.exp()
        H.extend((-torch.where(p > 0, p * logp, 0).sum(-1)).cpu().tolist())
    del logits
    return dict(status='ok', braco='humano', tipo='humano', delta=None, tau=None, amostra=None,
                prompt_tokens=len(p_ids), ids=r_ids, text=ctx.decode(r_ids),
                **anotar(ctx, p_ids[-1], r_ids, H),
                candidato=None, eligible_stone=None, eligible_hibrido=None, stop='referencia',
                max_new_tokens=None, corte=corte(task, r_ids, ctx.decode, ctx.especiais),
                segundos=round(time.perf_counter() - t0, 3))


# ------------------------------------------------------------------------ main
def configurar_log(pasta):
    pasta.mkdir(parents=True, exist_ok=True)
    arq = pasta / f'gerar_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}.log'
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s', '%Y-%m-%d %H:%M:%S')
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(arq, encoding='utf-8')):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.INFO)
    return arq


def versoes():
    out = {'python': platform.python_version()}
    for p in ('torch', 'transformers', 'accelerate', 'tokenizers', 'numpy', 'pandas', 'pyarrow', 'PyYAML'):
        try:
            out[p] = im.version(p)
        except im.PackageNotFoundError:
            out[p] = None
    return out


def checkpoint(run, t, braco, amostra=None):
    nome = f"{t['key']}_humano" if braco == 'humano' else f"{t['key']}_{braco}_s{amostra}"
    return run / t['dataset'] / t['split'] / f'{nome}.json.gz'


def registrar_bracos(run, cfg, n):
    """bracos.json acumula o que já foi pedido nesta pasta (analisar.py lê daqui)."""
    path = run / 'bracos.json'
    atual = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    novo = dict(bracos=sorted(set(atual.get('bracos', [])) | {b['nome'] for b in comum.bracos(cfg)}
                              | ({'humano'} if 'humano' in cfg['arms'] else set())),
                deltas=sorted(set(atual.get('deltas', [])) | set(cfg['deltas'])),
                taus=sorted(set(atual.get('taus', [])) | set(cfg['taus'])),
                n_amostras=max(atual.get('n_amostras', 0), n))
    write_text_atomic(path, json.dumps(novo, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--inicio', type=int, default=None, help='índice inicial da fatia (inclusivo)')
    ap.add_argument('--fim', type=int, default=None, help='índice final da fatia (exclusivo)')
    ap.add_argument('--sorteio', type=int, default=None, help='amostrar N tarefas antes de fatiar (padrão: YAML)')
    ap.add_argument('--sorteio-seed', type=int, default=None, help='padrão: sorteio_seed ou seed do YAML')
    ap.add_argument('--limite', type=int, default=None, help='sobrescreve limit de cada conjunto/divisão')
    ap.add_argument('--n_amostras', '--n-amostras', dest='n_amostras', type=int, default=None,
                    help='respostas por tarefa e braço (padrão: n_samples do YAML)')
    ap.add_argument('--listar', action='store_true', help='só mostra a seleção; não carrega o modelo')
    args = ap.parse_args(argv)
    cfg = comum.carregar_config(args.config)
    n_amostras = args.n_amostras or cfg['n_samples']
    if n_amostras < 1:
        ap.error('--n_amostras deve ser >= 1')

    configurar_log(comum.RESULTADOS / 'logs')
    log.info('Config %s: %s', args.config, json.dumps(cfg, ensure_ascii=False))
    log.info('Versões: %s', versoes())

    hf_cfg = AutoConfig.from_pretrained(cfg['model_id'], revision=cfg['model_revision'])
    revision = hf_cfg._commit_hash or cfg['model_revision']
    log.info('Modelo %s, revisão pedida %s, resolvida %s', cfg['model_id'], cfg['model_revision'], revision)
    tok = AutoTokenizer.from_pretrained(cfg['model_id'], revision=revision)

    tasks, grupos, inv = inventario(cfg, tok, revision)
    for (ds, sp), ts in grupos.items():
        ref = np.array([t['ref_tokens'] for t in ts if t['ref_tokens'] is not None], dtype=float)
        p = np.percentile(ref, [50, 90, 99, 100]) if len(ref) else [np.nan] * 4
        fmts = pd.Series([comum.formato(t) for t in ts]).value_counts().to_dict()
        log.info('Inventário %s/%s: %d tarefas; formatos %s; sem referência humana %d; '
                 'tokens da referência p50=%.0f p90=%.0f p99=%.0f máx=%.0f', ds, sp, len(ts), fmts,
                 sum(not t.get('referencia') for t in ts), *p)
    sel = comum.selecionar(grupos, cfg, args.inicio, args.fim, args.sorteio, args.sorteio_seed, args.limite)
    log.info('Seleção: %d tarefas (inicio=%s fim=%s sorteio=%s sorteio_seed=%s limite=%s); amostras: %d',
             len(sel), args.inicio, args.fim, args.sorteio if args.sorteio is not None else cfg['sorteio'],
             args.sorteio_seed, args.limite, n_amostras)
    if args.listar:
        for t in sel:
            print(f"{t['dataset']}\t{t['split']}\t{t['id']}\t{t['key']}")
        return 0

    nome_ds = {CONJUNTOS[n]['rotulo']: n for n in cfg['datasets']}
    max_new = {n: d['max_new_tokens'] for n, d in cfg['datasets'].items()}
    # Braços, deltas, taus e amostras ficam fora do hash: acrescentá-los reaproveita a pasta.
    run_settings = dict(protocol=PROTOCOLO, sources={f: FONTES[f] for n in cfg['datasets'] for f in CONJUNTOS[n]['fontes']},
                        inventory=digest(tasks), model=cfg['model_id'], revision=revision, dtype=cfg['dtype'],
                        gamma=cfg['gamma'], hash_key=cfg['hash_key'],
                        temperature=cfg['temperature'], top_k=cfg['top_k'], seed=cfg['seed'],
                        max_new_tokens=max_new, stop_check_every=cfg['stop_check_every'],
                        suppress_tokens=cfg['suppress_tokens'], transformers=im.version('transformers'))
    run = comum.RESULTADOS / f"{cfg['name']}_{digest(run_settings)[:14]}"
    for ds, sp in grupos:
        (run / ds / sp).mkdir(parents=True, exist_ok=True)
    if (run / 'config.json.gz').exists():
        assert read_gz(run / 'config.json.gz') == json.loads(json.dumps(run_settings)), 'config.json.gz diverge'
    else:
        write_gz(run / 'config.json.gz', run_settings)
    write_text_atomic(run / 'inventario.csv', inv.to_csv(index=False))
    registrar_bracos(run, cfg, n_amostras)
    log.info('Pasta da rodada: %s', run)

    gerados = comum.bracos(cfg)
    humano = 'humano' in cfg['arms']

    def trabalhos(t):
        out = [('humano', None)] if humano else []
        return out + [(b['nome'], s) for s in range(n_amostras) for b in gerados]

    pend_por_grupo = {}
    for t in sel:
        if any(not checkpoint(run, t, b, s).exists() for b, s in trabalhos(t)):
            pend_por_grupo.setdefault((t['dataset'], t['split']), []).append(t)
    # Alterna os conjuntos, como no notebook, para ter pares de cada um logo no início.
    fila = [t for linha in zip_longest(*pend_por_grupo.values()) for t in linha if t is not None]
    log.info('Braços: %s; por tarefa: %d arquivos. Pendentes: %d de %d tarefas selecionadas',
             (['humano'] if humano else []) + [b['nome'] for b in gerados], len(trabalhos(sel[0])) if sel else 0,
             len(fila), len(sel))
    if not fila:
        return 0

    if cfg['device'] == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('device: cuda, mas não há GPU visível.')
        device_map = {'': 0}
        log.info('GPU: %s', torch.cuda.get_device_name(0))
    else:
        device_map = {'': 'cpu'}
    model = AutoModelForCausalLM.from_pretrained(cfg['model_id'], revision=revision, dtype=getattr(torch, cfg['dtype']),
                                                 device_map=device_map, attn_implementation='sdpa').eval()
    if cfg['device'] == 'cuda':
        log.info('VRAM após carregar: %.2f GB', torch.cuda.memory_allocated(0) / 2**30)

    vocab_logits = model.get_output_embeddings().weight.shape[0]
    log.info('Vocabulário: logits=%d, config.vocab_size=%d, tokenizador len=%d (base %d)',
             vocab_logits, model.config.vocab_size, len(tok), tok.vocab_size)
    if vocab_logits != model.config.vocab_size:
        raise RuntimeError(f'lm_head tem {vocab_logits} linhas, config.vocab_size={model.config.vocab_size}.')
    if len(tok) > vocab_logits:
        raise RuntimeError(f'Tokenizador ({len(tok)}) maior que os logits ({vocab_logits}).')

    suprimir = []
    for s in cfg['suppress_tokens']:
        tid = tok.convert_tokens_to_ids(s)
        if tid is None or tid == tok.unk_token_id or tok.convert_ids_to_tokens(tid) != s:
            raise RuntimeError(f'Token a suprimir não existe no tokenizador: {s}')
        suprimir.append(tid)
    log.info('Tokens suprimidos: %s', dict(zip(cfg['suppress_tokens'], suprimir)))
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [tok.eos_token_id])
    eos = [x for x in eos if x is not None]
    if set(eos) & set(suprimir):
        raise RuntimeError('Um token de EOS está na lista de supressão.')
    ctx = Contexto(cfg, tok, model, vocab_logits, suprimir, eos)   # lista verde = vocabulário dos logits
    log.info('EOS: %s; contexto do modelo: %d; max_new_tokens: %s; verificação a cada %d tokens',
             eos, ctx.context, max_new, cfg['stop_check_every'])

    por_nome = {b['nome']: b for b in gerados}
    falhas = 0
    for i, t in enumerate(fila, 1):
        for nome, amostra in trabalhos(t):
            dest = checkpoint(run, t, nome, amostra)
            if dest.exists():
                continue
            try:
                if nome == 'humano':
                    s = pontuar_humano(ctx, t)
                else:
                    s = generate(ctx, t, por_nome[nome], amostra, max_new[nome_ds[t['dataset']]])
                write_gz(dest, s)
                if s['status'] == 'ok':
                    log.info('[%d/%d] %s/%s %s %-22s s%s %4d tokens %6.1fs parada=%s corte=%s ast=%s',
                             i, len(fila), t['dataset'], t['split'], t['id'], nome,
                             '-' if amostra is None else amostra, len(s['ids']), s['segundos'], s['stop'],
                             s['corte']['tokens'], s['corte']['ast_ok'])
                else:
                    log.warning('[%d/%d] %s %s %s: %s', i, len(fila), t['dataset'], t['id'], nome, s['status'])
            except torch.OutOfMemoryError:
                falhas += 1
                log.error('Memória insuficiente; continua pendente: %s %s %s s%s', t['dataset'], t['id'], nome, amostra)
                gc.collect()
                torch.cuda.empty_cache()
    log.info('Fim. Falhas por memória: %d', falhas)
    return 2 if falhas else 0


if __name__ == '__main__':
    sys.exit(main())
