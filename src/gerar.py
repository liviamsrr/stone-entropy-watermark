"""Geração STONE / sem marca por tarefa: versão em script do notebook v5.

    python src/gerar.py --config configs/smoke.yaml
    python src/gerar.py --config configs/apps.yaml --sorteio 600 --inicio 0 --fim 200
    python src/gerar.py --config configs/apps.yaml --listar     # só mostra a seleção

Cada (tarefa, braço) vira um resultados/<rodada>/<conjunto>/<divisão>/<chave>_<braço>.json.gz,
gravado de forma atômica. Rodar de novo com a mesma configuração retoma a rodada.
"""
import argparse
import functools
import gc
import importlib.metadata as im
import json
import logging
import math
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
from comum import (CONJUNTOS, DADOS, FONTES, PARADA, RESULTADOS, corte, digest, fetch,  # noqa: E402
                   fronteira, read_gz, write_gz, write_text_atomic)

PROTOCOLO = 'scripts-1'   # o notebook v5 usa 'v5.1'; aqui mudam parada e max_new_tokens
BRACOS = ('sem_marca', 'stone')
SYN = set(('True False None and as assert async await break class continue def del elif else '
           'except finally for from global if import in is lambda nonlocal not or pass raise return '
           'try while with yield int float complex str bytes bool list tuple set dict NoneType').split())
PUNCT = set('()[]{},:.;@->=+*/%<>!&|^~')
log = logging.getLogger('gerar')


# ------------------------------------------------------------------ inventário
def apps_task(d, split, tok):
    raw = d.get('input_output') or {}
    # Só precisamos de fn_name. Entradas dos testes podem ter inteiros de milhares de dígitos.
    io_ = json.loads(raw, parse_int=str) if isinstance(raw, str) else raw
    fn = (io_ or {}).get('fn_name')
    if not fn:
        return None
    q = d['question']
    starter = d.get('starter_code') or ''
    reference = d.get('solutions') or []
    if isinstance(reference, str):
        reference = json.loads(reference)
    codes = [s for s in (reference or []) if isinstance(s, str) and s.strip()]
    lens = []
    for i in range(0, len(codes), 32):
        batch = tok(codes[i:i + 32], add_special_tokens=False, truncation=False, verbose=False)
        lens.extend(map(len, batch.input_ids))
    prompt = f"\nQUESTION:\n{q}\n{starter}\nUse Call-Based Format\n\nANSWER:\n"
    identifier = d['id'] if 'id' in d else d['problem_id']
    return dict(dataset='APPS', split=split, id=str(identifier), prompt=prompt,
                ref_tokens=float(np.median(lens)) if lens else None,
                difficulty=d.get('difficulty') or '', target=fn)


def carregar_conjunto(nome, tok, model_id, revision):
    """Inventário de um conjunto, em cache em dados/ (depende do tokenizador)."""
    import gzip
    info = CONJUNTOS[nome]
    chave = digest(dict(sources={f: FONTES[f] for f in info['fontes']}, model=model_id,
                        revision=revision, schema=1))[:12]
    index = DADOS / f'inventario_{nome}_{chave}.json.gz'
    if index.exists():
        return read_gz(index)
    ntok = lambda s: len(tok.encode(s, add_special_tokens=False, verbose=False))  # noqa: E731
    tasks = []
    if nome == 'humaneval':
        with gzip.open(fetch('HumanEval+', log=log.info), 'rt', encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                tasks.append(dict(dataset='HumanEval+', split='test', id=d['task_id'],
                                  prompt=d['prompt'], ref_tokens=ntok(d['canonical_solution']),
                                  difficulty='', target=d['entry_point']))
        assert len(tasks) == 164, f'HumanEval+: esperados 164, encontrados {len(tasks)}'
    elif nome == 'apps':
        for split in ('test', 'train'):
            n = 0
            with fetch(f'APPS/{split}', log=log.info).open(encoding='utf-8') as f:
                for line in f:
                    n += 1
                    t = apps_task(json.loads(line), split, tok)
                    if t is not None:
                        tasks.append(t)
            if n != 5000:
                raise ValueError(f'APPS {split}: esperados 5000 registros, encontrados {n}.')
    elif nome == 'classeval':
        cols = ['task_id', 'skeleton', 'solution_code', 'class_name']
        for d in pd.read_parquet(fetch('ClassEval', log=log.info), columns=cols).to_dict('records'):
            prompt = ('Implement the complete Python class described by the following skeleton. '
                      'Return the entire class as Python code.\n\n' + d['skeleton'] + '\n\n# Implementation\n')
            tasks.append(dict(dataset='ClassEval', split='test', id=d['task_id'], prompt=prompt,
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
    inv = pd.DataFrame(tasks).drop(columns='prompt')
    inv['faixa_ref'] = ''
    for _, frame in inv.groupby(['dataset', 'split'], sort=False):
        q1, q2 = frame.ref_tokens.quantile([1 / 3, 2 / 3])
        x = frame.ref_tokens
        inv.loc[frame.index, 'faixa_ref'] = np.select(
            [x.isna(), x <= q1, x <= q2], ['sem referência', 'curta', 'média'], default='longa')
    return tasks, grupos, inv


# --------------------------------------------------------- marca e parada
class TraceSTONE(LogitsProcessor):
    """Mesmo processador do notebook v5, com duas mudanças:
    - tokens de preenchimento (FIM) recebem -inf antes de tudo, inclusive da entropia;
    - falha se o vocabulário dos logits difere do usado na lista verde."""

    def __init__(self, ctx, marked, seed, device):
        self.ctx, self.marked = ctx, marked
        self.probe = torch.Generator(device=device).manual_seed(seed)
        self.entropies, self.eligible = [], []

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
        self.entropies.append(-torch.where(p > 0, p * logp, 0).sum())
        shortlist, ids = torch.topk(values / c.cfg['temperature'], min(c.cfg['top_k'], values.numel()))
        sample = torch.multinomial(shortlist.softmax(-1), 1, generator=self.probe)
        eligible = not c.syntactic(int(ids[sample].item()))
        self.eligible.append(eligible)
        if self.marked and eligible:
            prev = int(input_ids[0, -1].item())
            scores = scores.clone()
            scores[0] += c.cfg['delta'] * c.mask_cpu(prev).to(scores.device, dtype=scores.dtype)
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


@torch.inference_mode()
def generate(ctx, task, arm, max_new):
    cfg, tok, model = ctx.cfg, ctx.tok, ctx.model
    enc = tok(task['prompt'], return_tensors='pt', add_special_tokens=False,
              return_token_type_ids=False, truncation=False)
    plen = enc.input_ids.shape[1]
    if not 0 < plen < ctx.context:
        return dict(status='prompt_sem_espaco_no_contexto', prompt_tokens=plen)
    max_efetivo = min(max_new, ctx.context - plen)
    seed = int(digest([cfg['seed'], task['key']])[:8], 16)   # igual nos dois braços (pareado)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    enc = enc.to(model.device)
    proc = TraceSTONE(ctx, arm == 'stone', seed ^ 0xA19D, model.device)
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
    last = int(enc.input_ids[0, -1].item())
    counted, green, special = [], [], []
    for tid in ids:
        sp = tid in ctx.especiais
        special.append(sp)
        counted.append(not sp and not ctx.syntactic(tid))
        green.append(bool(ctx.mask_cpu(last)[tid]) if not sp else False)
        last = tid
    if ids and ids[-1] in ctx.eos:
        stop = 'eos'
    elif parada.disparou:
        stop = PARADA[task['dataset']]
    elif len(ids) >= max_efetivo:
        stop = 'max_new_tokens' if max_efetivo == max_new else 'contexto'
    else:
        stop = 'outro'
    return dict(status='ok', prompt_tokens=plen, ids=ids, text=ctx.decode(ids),
                H=h, eligible=proc.eligible, counted=counted, green=green, special=special, stop=stop,
                max_new_tokens=max_efetivo, corte=corte(task, ids, ctx.decode, ctx.especiais),
                segundos=round(segundos, 3))


# ------------------------------------------------------------------------ main
def configurar_log(pasta):
    pasta.mkdir(parents=True, exist_ok=True)
    arq = pasta / f'gerar_{datetime.now():%Y%m%d_%H%M%S}_{__import__("os").getpid()}.log'
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--inicio', type=int, default=None, help='índice inicial da fatia (inclusivo)')
    ap.add_argument('--fim', type=int, default=None, help='índice final da fatia (exclusivo)')
    ap.add_argument('--sorteio', type=int, default=None, help='amostrar N tarefas antes de fatiar')
    ap.add_argument('--sorteio-seed', type=int, default=None, help='padrão: seed do YAML')
    ap.add_argument('--limite', type=int, default=None, help='sobrescreve limit de cada conjunto/divisão')
    ap.add_argument('--listar', action='store_true', help='só mostra a seleção; não carrega o modelo')
    args = ap.parse_args(argv)
    cfg = comum.carregar_config(args.config)

    configurar_log(RESULTADOS / 'logs')
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
        log.info('Inventário %s/%s: %d tarefas; tokens da referência p50=%.0f p90=%.0f p99=%.0f máx=%.0f',
                 ds, sp, len(ts), *p)
    sel = comum.selecionar(grupos, cfg, args.inicio, args.fim, args.sorteio, args.sorteio_seed, args.limite)
    log.info('Seleção: %d tarefas (inicio=%s fim=%s sorteio=%s sorteio_seed=%s limite=%s)', len(sel),
             args.inicio, args.fim, args.sorteio, args.sorteio_seed, args.limite)
    if args.listar:
        for t in sel:
            print(f"{t['dataset']}\t{t['split']}\t{t['id']}\t{t['key']}")
        return 0

    nome_ds = {CONJUNTOS[n]['rotulo']: n for n in cfg['datasets']}
    max_new = {n: d['max_new_tokens'] for n, d in cfg['datasets'].items()}
    run_settings = dict(protocol=PROTOCOLO, sources={f: FONTES[f] for n in cfg['datasets'] for f in CONJUNTOS[n]['fontes']},
                        inventory=digest(tasks), model=cfg['model_id'], revision=revision, dtype=cfg['dtype'],
                        gamma=cfg['gamma'], delta=cfg['delta'], hash_key=cfg['hash_key'],
                        temperature=cfg['temperature'], top_k=cfg['top_k'], seed=cfg['seed'],
                        max_new_tokens=max_new, stop_check_every=cfg['stop_check_every'],
                        suppress_tokens=cfg['suppress_tokens'], transformers=im.version('transformers'))
    run = RESULTADOS / f"{cfg['name']}_{digest(run_settings)[:14]}"
    for ds, sp in grupos:
        (run / ds / sp).mkdir(parents=True, exist_ok=True)
    if (run / 'config.json.gz').exists():
        assert read_gz(run / 'config.json.gz') == json.loads(json.dumps(run_settings)), 'config.json.gz diverge'
    else:
        write_gz(run / 'config.json.gz', run_settings)
    write_text_atomic(run / 'inventario.csv', inv.to_csv(index=False))
    log.info('Pasta da rodada: %s', run)

    def checkpoint(t, arm):
        return run / t['dataset'] / t['split'] / f"{t['key']}_{arm}.json.gz"

    pend_por_grupo = {}
    for t in sel:
        if any(not checkpoint(t, a).exists() for a in BRACOS):
            pend_por_grupo.setdefault((t['dataset'], t['split']), []).append(t)
    # Alterna os conjuntos, como no notebook, para ter pares de cada um logo no início.
    fila = [t for linha in zip_longest(*pend_por_grupo.values()) for t in linha if t is not None]
    log.info('Pendentes: %d de %d tarefas selecionadas', len(fila), len(sel))
    if not fila:
        return 0

    if cfg['device'] == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('device: cuda, mas não há GPU visível.')
        device_map = {'': 0}
        log.info('GPU: %s', torch.cuda.get_device_name(0))
    else:
        device_map = {'': 'cpu'}
    dtype = getattr(torch, cfg['dtype'])
    model = AutoModelForCausalLM.from_pretrained(cfg['model_id'], revision=revision, dtype=dtype,
                                                 device_map=device_map, attn_implementation='sdpa').eval()
    if torch.cuda.is_available() and cfg['device'] == 'cuda':
        log.info('VRAM após carregar: %.2f GB', torch.cuda.memory_allocated(0) / 2**30)

    vocab_logits = model.get_output_embeddings().weight.shape[0]
    log.info('Vocabulário: logits=%d, config.vocab_size=%d, tokenizador len=%d (base %d)',
             vocab_logits, model.config.vocab_size, len(tok), tok.vocab_size)
    if vocab_logits != model.config.vocab_size:
        raise RuntimeError(f'lm_head tem {vocab_logits} linhas, config.vocab_size={model.config.vocab_size}.')
    if len(tok) > vocab_logits:
        raise RuntimeError(f'Tokenizador ({len(tok)}) maior que os logits ({vocab_logits}).')
    vocab_verde = vocab_logits    # a lista verde usa exatamente o vocabulário dos logits

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
    ctx = Contexto(cfg, tok, model, vocab_verde, suprimir, eos)
    log.info('EOS: %s; contexto do modelo: %d; max_new_tokens: %s; verificação a cada %d tokens',
             eos, ctx.context, max_new, cfg['stop_check_every'])

    falhas = 0
    for i, t in enumerate(fila, 1):
        for arm in BRACOS:
            dest = checkpoint(t, arm)
            if dest.exists():
                continue
            try:
                s = generate(ctx, t, arm, max_new[nome_ds[t['dataset']]])
                write_gz(dest, s)
                if s['status'] == 'ok':
                    log.info('[%d/%d] %s/%s %s %-9s %4d tokens %6.1fs parada=%s corte=%s',
                             i, len(fila), t['dataset'], t['split'], t['id'], arm, len(s['ids']),
                             s['segundos'], s['stop'], s['corte']['tokens'])
                else:
                    log.warning('[%d/%d] %s %s %s: %s', i, len(fila), t['dataset'], t['id'], arm, s['status'])
            except torch.OutOfMemoryError:
                falhas += 1
                log.error('Memória insuficiente; continua pendente: %s %s %s', t['dataset'], t['id'], arm)
                gc.collect()
                torch.cuda.empty_cache()
    log.info('Fim. Falhas por memória: %d', falhas)
    return 2 if falhas else 0


if __name__ == '__main__':
    sys.exit(main())
