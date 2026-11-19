"""Partes compartilhadas por gerar.py e baixar_dados.py.

Este módulo usa apenas a biblioteca padrão (o YAML é importado só ao ler a
configuração), para que baixar_dados.py rode na sessão de login sem torch.
"""
import ast
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import time
import token
import tokenize
from pathlib import Path
from urllib.request import Request, urlopen

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / 'dados'
RESULTADOS = RAIZ / 'resultados'

# Mesmas URLs (com commits fixos) e mesmos nomes de arquivo do notebook v5.
FONTES = {
    'HumanEval+': 'https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz',
    'APPS/test': 'https://huggingface.co/datasets/codeparrot/apps/resolve/21e74dd/test.jsonl',
    'APPS/train': 'https://huggingface.co/datasets/codeparrot/apps/resolve/21e74dd/train.jsonl',
    'ClassEval': 'https://huggingface.co/datasets/FudanSELab/ClassEval/resolve/fef204b/data/test-00000-of-00001-5c45fa6e45572491.parquet',
}
ARQUIVOS = {
    'HumanEval+': 'humaneval_v0.1.10.jsonl.gz',
    'APPS/test': 'apps_21e74dd_test.jsonl',
    'APPS/train': 'apps_21e74dd_train.jsonl',
    'ClassEval': 'classeval_fef204b.parquet',
}
# Nome do conjunto no YAML -> fontes, rótulo usado nos resultados e divisões.
CONJUNTOS = {
    'humaneval': dict(rotulo='HumanEval+', fontes=['HumanEval+'], divisoes=['test']),
    'apps': dict(rotulo='APPS', fontes=['APPS/test', 'APPS/train'], divisoes=['test', 'train']),
    'classeval': dict(rotulo='ClassEval', fontes=['ClassEval'], divisoes=['test']),
}
# Critério de parada por conjunto (item 4) e o motivo registrado quando ele dispara.
PARADA = {'HumanEval+': 'fim_da_funcao', 'APPS': 'fim_da_funcao', 'ClassEval': 'fim_da_classe'}


# --------------------------------------------------------------------- arquivos
def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _tmp(path):
    # Nome temporário único: jobs paralelos (fatias do APPS) podem gravar o mesmo arquivo.
    return path.with_name(f'{path.name}.{os.getpid()}.{time.time_ns()}.tmp')


def write_gz(path, value):
    """Gravação atômica: o arquivo final só aparece completo."""
    path = Path(path)
    tmp = _tmp(path)
    with gzip.open(tmp, 'wt', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    os.replace(tmp, path)


def read_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def write_text_atomic(path, text):
    path = Path(path)
    tmp = _tmp(path)
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)


def sha256_arquivo(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for bloco in iter(lambda: f.read(1 << 20), b''):
            h.update(bloco)
    return h.hexdigest()


def fetch(fonte, destino=DADOS, log=print):
    """Baixa a fonte para dados/ se o arquivo ainda não existe; devolve o caminho."""
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    path = destino / ARQUIVOS[fonte]
    if path.exists() and path.stat().st_size > 0:
        return path
    url = FONTES[fonte]
    log(f'Baixando {fonte} -> {path}')
    tmp = path.with_name(f'{path.name}.{os.getpid()}.part')
    for tentativa in range(3):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'stone-entropy-watermark'}), timeout=90) as r:
                with tmp.open('wb') as f:
                    shutil.copyfileobj(r, f, length=1 << 20)
            if tmp.stat().st_size == 0:
                raise OSError('download vazio')
            os.replace(tmp, path)
            return path
        except (OSError, TimeoutError) as e:
            tmp.unlink(missing_ok=True)
            if tentativa == 2:
                raise RuntimeError(f'Falha ao baixar {url}: {e}. Rode src/baixar_dados.py '
                                   'na sessão de login e tente de novo.') from e
            time.sleep(2 * (tentativa + 1))


# ---------------------------------------------------------------- configuração
PADRAO = dict(
    name='rodada', model_id='Qwen/Qwen2.5-Coder-7B', model_revision='main',
    dtype='float16', device='cuda',
    gamma=0.5, delta=0.5, hash_key=15485863, temperature=1.0, top_k=50, seed=0,
    stop_check_every=16,
    suppress_tokens=['<|fim_prefix|>', '<|fim_middle|>', '<|fim_suffix|>'],
)


def carregar_config(path):
    import yaml
    with open(path, encoding='utf-8') as f:
        bruto = yaml.safe_load(f) or {}
    desconhecidas = set(bruto) - set(PADRAO) - {'datasets'}
    if desconhecidas:
        raise ValueError(f'Chaves desconhecidas em {path}: {sorted(desconhecidas)}')
    cfg = {**PADRAO, **bruto}
    ds = cfg.get('datasets') or {}
    if not ds:
        raise ValueError('A configuração precisa de pelo menos um conjunto em "datasets".')
    for nome, d in ds.items():
        if nome not in CONJUNTOS:
            raise ValueError(f'Conjunto desconhecido: {nome}. Use {sorted(CONJUNTOS)}.')
        d = d or {}
        extra = set(d) - {'max_new_tokens', 'limit'}
        if extra:
            raise ValueError(f'Chaves desconhecidas em datasets.{nome}: {sorted(extra)}')
        if not isinstance(d.get('max_new_tokens'), int) or d['max_new_tokens'] < 1:
            raise ValueError(f'datasets.{nome}.max_new_tokens deve ser inteiro positivo.')
        if d.get('limit') is not None and (not isinstance(d['limit'], int) or d['limit'] < 1):
            raise ValueError(f'datasets.{nome}.limit deve ser inteiro positivo ou null.')
        ds[nome] = d
    if not (0 < cfg['gamma'] < 1 and cfg['delta'] >= 0 and cfg['top_k'] >= 1
            and cfg['temperature'] > 0 and cfg['stop_check_every'] >= 1):
        raise ValueError('Parâmetros fora do intervalo: 0<gamma<1, delta>=0, top_k>=1, '
                         'temperature>0, stop_check_every>=1.')
    if cfg['dtype'] not in {'float16', 'bfloat16', 'float32'} or cfg['device'] not in {'cuda', 'cpu'}:
        raise ValueError('dtype deve ser float16/bfloat16/float32 e device, cuda/cpu.')
    return cfg


def selecionar(grupos, cfg, inicio=None, fim=None, sorteio=None, sorteio_seed=None, limite=None):
    """Seleção determinística das tarefas.

    1. em cada (conjunto, divisão), as primeiras `limit` tarefas (YAML ou --limite);
    2. concatenação na ordem do YAML (APPS: test e depois train);
    3. --sorteio N: amostra N tarefas com a semente, mantendo a ordem do inventário;
    4. --inicio/--fim: fatia [inicio, fim) da lista resultante.
    A mesma semente dá a mesma amostra em todos os jobs, então fatias de um mesmo
    sorteio não se sobrepõem.
    """
    import random
    lista = []
    for nome, d in cfg['datasets'].items():
        info = CONJUNTOS[nome]
        lim = limite if limite is not None else d.get('limit')
        for divisao in info['divisoes']:
            ts = grupos[info['rotulo'], divisao]
            lista.extend(ts[:lim] if lim else ts)
    if sorteio is not None:
        semente = cfg['seed'] if sorteio_seed is None else sorteio_seed
        idx = sorted(random.Random(semente).sample(range(len(lista)), min(sorteio, len(lista))))
        lista = [lista[i] for i in idx]
    return lista[inicio:fim]


# ------------------------------------------------------- fronteiras (do v4)
_IGNORADOS = {tokenize.ENCODING, token.INDENT, token.DEDENT, token.NEWLINE,
              tokenize.NL, token.ENDMARKER, tokenize.COMMENT}
_ERROS = (tokenize.TokenError, IndentationError, SyntaxError)


def _offsets(texto):
    # Mesma divisão de linhas que o tokenize usa (readline de StringIO).
    offs = [0]
    for linha in io.StringIO(texto).readlines():
        offs.append(offs[-1] + len(linha))
    return offs


def _top_level(item, depth):
    return (item.start[1] == 0 and depth == 0 and item.type not in _IGNORADOS
            and item.string.strip())


def fronteira_continuacao(prompt, completion):
    """HumanEval+: lógica do notebook v4 (python_boundary).

    O prompt termina dentro da função; a função acaba no próximo elemento no
    nível zero de indentação, fora de parênteses, que comece depois do prompt.
    Devolve o índice (em caracteres da completion) ou None.
    """
    full = prompt + completion
    offs = _offsets(full)
    depth = 0
    try:
        for item in tokenize.generate_tokens(io.StringIO(full).readline):
            row, col = item.start
            absoluto = offs[min(row - 1, len(offs) - 1)] + col
            if absoluto >= len(prompt) and _top_level(item, depth):
                return max(0, absoluto - len(prompt))
            if item.type == token.OP:
                if item.string in '([{':
                    depth += 1
                elif item.string in ')]}':
                    depth = max(0, depth - 1)
    except _ERROS:
        # Uma saída truncada ainda pode ter uma fronteira anterior ao erro.
        pass
    return None


_CERCA_INICIO = re.compile(r'[ \t]*\n*[ \t]*```[^\n]*\n')
_CERCA_FIM = re.compile(r'^```', re.M)
_INICIO_CODIGO = re.compile(r'^(?:def|class|async\s+def|import|from|@)\b|^@', re.M)


def fronteira_bloco_alvo(completion, alvo):
    """APPS (alvo = fn_name) e ClassEval (alvo = nome da classe).

    O prompt não é Python, então a completion é analisada sozinha. A mesma regra
    do v4 é aplicada a partir do bloco de nível zero que define `alvo`
    (`def alvo` ou `class alvo`, em qualquer profundidade; ex.: método dentro de
    `class Solution`): a fronteira é o próximo elemento de nível zero depois dele.
    Imports e funções auxiliares antes do alvo não encerram a geração.
    Uma cerca de markdown (```) que abre a resposta é ignorada, e uma cerca de
    fechamento depois do alvo também conta como fronteira.
    """
    m = _CERCA_INICIO.match(completion)
    base = m.end() if m else 0
    # Prosa antes do código (ex.: "Here's the code:") quebraria o tokenize;
    # a análise começa na primeira linha de código no nível zero.
    codigo = _INICIO_CODIGO.search(completion, base)
    if codigo:
        base = codigo.start()
    corpo = completion[base:]
    cerca = _CERCA_FIM.search(corpo)
    trecho = corpo[:cerca.start()] if cerca else corpo
    offs = _offsets(trecho)
    depth, tem_alvo, anterior = 0, False, None
    try:
        for item in tokenize.generate_tokens(io.StringIO(trecho).readline):
            row, col = item.start
            if _top_level(item, depth) and tem_alvo:
                return base + offs[min(row - 1, len(offs) - 1)] + col
            if item.type == token.NAME and anterior in ('def', 'class') and item.string == alvo:
                tem_alvo = True
            if item.type == token.OP:
                if item.string in '([{':
                    depth += 1
                elif item.string in ')]}':
                    depth = max(0, depth - 1)
            if item.type not in _IGNORADOS:
                anterior = item.string if item.type == token.NAME else None
    except _ERROS:
        pass
    if cerca and tem_alvo:
        return base + cerca.start()
    return None


def fronteira(task, completion):
    if task['dataset'] == 'HumanEval+':
        return fronteira_continuacao(task['prompt'], completion)
    return fronteira_bloco_alvo(completion, task['target'])


def corte(task, ids, decode, especiais):
    """Corte pós-geração, como cut_function do v4: guarda a resposta integral e
    registra quantos tokens ficam até a fronteira. Se a fronteira cair dentro de
    um token do modelo, esse token é descartado (corte conservador)."""
    k_esp = next((i for i, t in enumerate(ids) if t in especiais), len(ids))
    texto = decode(ids[:k_esp])
    pos = fronteira(task, texto)
    if pos is None:
        manter, motivo = k_esp, 'sem_fronteira'
    else:
        manter, motivo = 0, PARADA[task['dataset']]
        for k in range(1, k_esp + 1):
            prefixo = decode(ids[:k])
            if len(prefixo) <= pos and texto.startswith(prefixo):
                manter = k
            elif len(prefixo) > pos:
                break
    mantido = decode(ids[:manter])
    codigo = task['prompt'] + mantido if task['dataset'] == 'HumanEval+' else mantido
    try:
        arvore = ast.parse(codigo)
        sintaxe_ok = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                         for n in arvore.body)
    except (SyntaxError, ValueError, TypeError):
        sintaxe_ok = False
    return dict(chars=pos, tokens=manter, motivo=motivo, sintaxe_ok=sintaxe_ok)
