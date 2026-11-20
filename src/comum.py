"""Partes compartilhadas por gerar.py, analisar.py, avaliar.py e baixar_dados.py.

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
# STONE_DADOS / STONE_RESULTADOS permitem testar sem tocar nas pastas reais.
DADOS = Path(os.environ.get('STONE_DADOS') or RAIZ / 'dados')
RESULTADOS = Path(os.environ.get('STONE_RESULTADOS') or RAIZ / 'resultados')

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
# Motivo registrado quando o critério de fim dispara.
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


def fetch(fonte, destino=None, log=print):
    """Baixa a fonte para dados/ se o arquivo ainda não existe; devolve o caminho."""
    destino = Path(destino or DADOS)
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


# --------------------------------------------------------------- rodadas
def chave_inventario(nome, model_id, revision):
    info = CONJUNTOS[nome]
    return digest(dict(sources={f: FONTES[f] for f in info['fontes']}, model=model_id,
                       revision=revision, schema=3))[:12]   # 3: ClassEval = skeleton + reimplementação


def localizar_rodada(rodada=None, config=None):
    """--rodada PASTA, ou --config YAML (a única pasta resultados/<name>_*)."""
    if rodada:
        run = Path(rodada)
    else:
        nome = carregar_config(config)['name']
        candidatas = sorted(p for p in RESULTADOS.glob(f'{nome}_*') if (p / 'config.json.gz').exists())
        if len(candidatas) != 1:
            raise SystemExit(f'Esperava uma pasta resultados/{nome}_*, achei {len(candidatas)}: '
                             f'{[p.name for p in candidatas]}. Use --rodada.')
        run = candidatas[0]
    if not (run / 'config.json.gz').exists():
        raise SystemExit(f'{run} não é uma pasta de rodada (falta config.json.gz).')
    return run


def tarefas_da_rodada(run):
    """Tarefas (com prompt, referência e testes) do inventário em cache em dados/."""
    cfg = read_gz(Path(run) / 'config.json.gz')
    nomes = {n for n, info in CONJUNTOS.items() if any(f in cfg['sources'] for f in info['fontes'])}
    tarefas = {}
    for nome in sorted(nomes):
        path = DADOS / f"inventario_{nome}_{chave_inventario(nome, cfg['model'], cfg['revision'])}.json.gz"
        if not path.exists():
            raise SystemExit(f'Inventário não encontrado: {path}. Rode gerar.py com a mesma configuração.')
        for t in read_gz(path):
            t['key'] = digest([t['dataset'], t['split'], t['id']])[:24]
            tarefas[t['key']] = t
    return cfg, tarefas


_ARQ = re.compile(r'^(?P<key>[0-9a-f]{24})_(?:(?P<humano>humano)|(?P<braco>.+)_s(?P<amostra>\d+))\.json\.gz$')


def respostas_da_rodada(run):
    """Gera (key, braço, amostra, caminho) para cada checkpoint da rodada."""
    for path in sorted(Path(run).glob('*/*/*.json.gz')):
        m = _ARQ.match(path.name)
        if m:
            if m['humano']:
                yield m['key'], 'humano', None, path
            else:
                yield m['key'], m['braco'], int(m['amostra']), path


# ---------------------------------------------------------------- configuração
PADRAO = dict(
    name='rodada', model_id='Qwen/Qwen2.5-Coder-7B', model_revision='main',
    dtype='float16', device='cuda',
    gamma=0.5, hash_key=15485863, temperature=1.0, top_k=50, seed=0,
    deltas=[0.5, 1.0],            # um braço stone e um hibrido por delta
    taus=[0.3, 0.9],              # limiares de H do braço híbrido (e regras do detector)
    arms=['sem_marca', 'stone', 'hibrido', 'humano'],
    n_samples=1,                  # respostas por tarefa (--n_amostras sobrescreve)
    stop_check_every=16,
    suppress_tokens=['<|fim_prefix|>', '<|fim_middle|>', '<|fim_suffix|>'],
    sorteio=None, sorteio_seed=None,   # padrões de --sorteio/--sorteio-seed
)
BRACOS_VALIDOS = {'sem_marca', 'stone', 'hibrido', 'humano'}


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
    if not (0 < cfg['gamma'] < 1 and cfg['top_k'] >= 1 and cfg['temperature'] > 0
            and cfg['stop_check_every'] >= 1 and cfg['n_samples'] >= 1):
        raise ValueError('Parâmetros fora do intervalo: 0<gamma<1, top_k>=1, temperature>0, '
                         'stop_check_every>=1, n_samples>=1.')
    if not cfg['deltas'] or any(d < 0 for d in cfg['deltas']) or any(t < 0 for t in cfg['taus']):
        raise ValueError('deltas precisa de ao menos um valor >= 0; taus, valores >= 0.')
    if 'hibrido' in cfg['arms'] and not cfg['taus']:
        raise ValueError('O braço hibrido precisa de ao menos um tau.')
    if set(cfg['arms']) - BRACOS_VALIDOS:
        raise ValueError(f'arms aceita {sorted(BRACOS_VALIDOS)}.')
    if cfg['dtype'] not in {'float16', 'bfloat16', 'float32'} or cfg['device'] not in {'cuda', 'cpu'}:
        raise ValueError('dtype deve ser float16/bfloat16/float32 e device, cuda/cpu.')
    cfg['deltas'] = [float(x) for x in cfg['deltas']]
    cfg['taus'] = [float(x) for x in cfg['taus']]
    return cfg


def nome_braco(tipo, delta=None, tau=None):
    """sem_marca, humano, stone_d0.5, hibrido_d0.5_t0.9."""
    if tipo in ('sem_marca', 'humano'):
        return tipo
    if tipo == 'stone':
        return f'stone_d{float(delta)}'
    return f'hibrido_d{float(delta)}_t{float(tau)}'


def bracos(cfg):
    """Braços gerados (com parâmetros), na ordem de execução. 'humano' fica à parte."""
    out = []
    if 'sem_marca' in cfg['arms']:
        out.append(dict(nome='sem_marca', tipo='sem_marca', delta=0.0, tau=None))
    if 'stone' in cfg['arms']:
        out += [dict(nome=nome_braco('stone', d), tipo='stone', delta=d, tau=None) for d in cfg['deltas']]
    if 'hibrido' in cfg['arms']:
        out += [dict(nome=nome_braco('hibrido', d, t), tipo='hibrido', delta=d, tau=t)
                for d in cfg['deltas'] for t in cfg['taus']]
    return out


def selecionar(grupos, cfg, inicio=None, fim=None, sorteio=None, sorteio_seed=None, limite=None):
    """Seleção determinística das tarefas.

    1. em cada (conjunto, divisão), as primeiras `limit` tarefas (YAML ou --limite);
    2. concatenação na ordem do YAML (APPS: test e depois train);
    3. sorteio N (YAML ou --sorteio): amostra N tarefas com a semente, mantendo a
       ordem do inventário;
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
    sorteio = cfg.get('sorteio') if sorteio is None else sorteio
    if sorteio:
        semente = sorteio_seed if sorteio_seed is not None else (
            cfg.get('sorteio_seed') if cfg.get('sorteio_seed') is not None else cfg['seed'])
        idx = sorted(random.Random(semente).sample(range(len(lista)), min(sorteio, len(lista))))
        lista = [lista[i] for i in idx]
    return lista[inicio:fim]


# --------------------------------------------------------------- ClassEval
_ASPAS_CURVAS = re.compile('[“”]{3}')
_ASPAS_EXTERNAS = re.compile(r"\s*'{3}\n(.*)\n'{3}\s*", re.S)


def _fechar_docstrings(texto):
    """Fecha uma docstring de método que ficou aberta até o próximo def/class (ClassEval_22)."""
    out, aberta, ind = [], False, ''
    for l in texto.split('\n'):
        if aberta and re.match(r'[ \t]*(?:async\s+)?(?:def|class)\s', l) and len(l) - len(l.lstrip()) < len(ind):
            while out and not out[-1].strip():
                out.pop()
            out += [ind + '"' * 3, '']
            aberta = False
        out.append(l)
        n = l.count('"' * 3)
        if not aberta and l.strip().startswith('"' * 3) and n == 1:
            aberta, ind = True, l[:len(l) - len(l.lstrip())]
        elif aberta and n:
            aberta = False
    return '\n'.join(out)


def classeval_skeleton(skeleton):
    """Corrige defeitos de digitação de 4 skeletons do ClassEval (fef204b) e devolve
    (texto, correções). Fora isso, o skeleton fica como no conjunto:
    - ClassEval_5: skeleton inteiro dentro de aspas simples triplas (vira uma string, sem classe);
    - ClassEval_43 e 46: aspas curvas triplas no lugar de aspas duplas triplas;
    - ClassEval_22: docstring de register_class sem fechamento."""
    s, correcoes = skeleton, []
    m = _ASPAS_EXTERNAS.fullmatch(s)
    if m:
        s, correcoes = m.group(1), correcoes + ['aspas_externas']
    if _ASPAS_CURVAS.search(s):
        s, correcoes = _ASPAS_CURVAS.sub('"' * 3, s), correcoes + ['aspas_curvas']
    if not compila(s):
        s2 = _fechar_docstrings(s)
        if compila(s2):
            s, correcoes = s2, correcoes + ['docstring_aberta']
    return s, correcoes


def classeval_prompt(skeleton):
    """Formato holístico do ClassEval para modelo base (Du et al., 2023: prompt = skeleton),
    sem nenhuma instrução em linguagem natural: imports, `class Nome:` com docstring, o
    construtor e as assinaturas dos métodos com docstrings. O prompt termina no corpo da
    classe, depois do último método; o modelo reescreve os métodos com corpo dentro da
    mesma classe (em Python vale a última definição). Mesmo prompt em todos os braços.
    Devolve (prompt, correções)."""
    s, correcoes = classeval_skeleton(skeleton)
    return s.rstrip() + '\n\n', correcoes


def classeval_referencia(solution_code, class_name, prompt):
    """Texto humano para o mesmo prompt: o corpo da classe da solution_code (construtor e
    métodos com implementação), na indentação do skeleton (4 espaços nas 100 tarefas).
    Código de nível zero fora da classe (imports e, no ClassEval_52, chamadas
    nltk.download) fica de fora. None se prompt + corpo não compilar."""
    try:
        arvore = ast.parse(solution_code)
    except SyntaxError:
        return None
    cls = next((n for n in arvore.body if isinstance(n, ast.ClassDef) and n.name == class_name), None)
    if cls is None or not cls.body:
        return None
    primeiro = cls.body[0]
    inicio = min([primeiro.lineno] + [d.lineno for d in getattr(primeiro, 'decorator_list', [])])
    linhas = solution_code.splitlines(keepends=True)
    corpo = ''.join(linhas[inicio - 1:cls.end_lineno])
    corpo = corpo if corpo.endswith('\n') else corpo + '\n'
    return corpo if compila(prompt + corpo) else None


# ------------------------------------------------------------------- APPS
def _assinatura_corpo(starter, fn):
    """Se o starter termina na assinatura de `fn`, devolve a indentação do corpo."""
    defs = list(re.finditer(rf'^([ \t]*)(?:async[ \t]+)?def[ \t]+{re.escape(fn)}\b', starter, re.M))
    if not defs:
        return None
    base = defs[-1].group(1)
    ind = base + ('\t' if '\t' in base else '    ')
    try:
        arvore = ast.parse(starter + '\n' + ind + 'pass\n')
    except SyntaxError:
        return None
    alvos = [n for n in ast.walk(arvore) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == fn]
    ultimo = max((n for n in ast.walk(arvore) if hasattr(n, 'lineno')), key=lambda n: (n.lineno, n.col_offset))
    if alvos and len(alvos[-1].body) == 1 and isinstance(alvos[-1].body[0], ast.Pass) and ultimo is alvos[-1].body[0]:
        return ind
    return None


def apps_prompt(question, starter, fn):
    """Formato de completar função, como no HumanEval+: enunciado em comentários e,
    em seguida, o starter_code terminando na assinatura de `fn`. O modelo completa
    só o corpo. Sem starter utilizável, o modelo escreve a função inteira ('bloco').
    Devolve (prompt, formato, indentação do corpo)."""
    linhas = ['# QUESTION:'] + [('# ' + l).rstrip() for l in question.strip().splitlines()]
    cab = '\n'.join(linhas + ['#', '# Use Call-Based Format', ''])
    st = (starter or '').rstrip()
    ind = _assinatura_corpo(st, fn) if st else None
    if ind is not None:
        return cab + st + '\n', 'continuacao', ind
    return cab, 'bloco', None


_IMPORTS = (ast.Import, ast.ImportFrom)


def apps_referencia(solutions, fn, prompt, formato, ind):
    """Primeira solução válida para o mesmo prompt.

    continuacao: o texto humano é tudo o que vem depois da assinatura de `fn` até o
    fim da solução (corpo, métodos e funções seguintes). Vale a solução em que, antes
    de `fn`, só há imports e, se `fn` é método, a linha da classe (`fn` é o primeiro
    membro). Os imports caem; o avaliador do APPS já importa os módulos comuns.
    A indentação do def na solução é trocada pela do starter, e prompt + texto
    precisa compilar.
    bloco: a solução inteira, se compilar e definir `fn`.
    """
    base_alvo = ind[:-1] if ind and ind.endswith('\t') else (ind[:-4] if ind else '')
    for sol in solutions:
        if not isinstance(sol, str) or not sol.strip():
            continue
        try:
            arvore = ast.parse(sol)
        except (SyntaxError, ValueError):
            continue
        alvos = [n for n in ast.walk(arvore) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == fn]
        if not alvos:
            continue
        if formato == 'bloco':
            return sol if sol.endswith('\n') else sol + '\n'
        node = alvos[0]
        antes_ok = True
        for n in arvore.body:
            if n is node:
                break
            if isinstance(n, _IMPORTS):
                continue
            if isinstance(n, ast.ClassDef) and n.body and n.body[0] is node:
                break
            antes_ok = False
            break
        if not antes_ok or node.body[0].lineno == node.lineno:
            continue
        linhas = sol.splitlines(keepends=True)
        sd = re.match(r'[ \t]*', linhas[node.lineno - 1]).group()
        if (sd == '') != (base_alvo == ''):
            continue        # método no starter e função solta na solução, ou o contrário
        corpo = ''
        for l in linhas[node.body[0].lineno - 1:]:
            corpo += base_alvo + l[len(sd):] if l.startswith(sd) and l.strip() else l
        if not corpo.endswith('\n'):
            corpo += '\n'
        try:
            ast.parse(prompt + corpo)
        except SyntaxError:
            continue
        return corpo
    return None


def apps_tarefa(d, split):
    """Registro do APPS -> tarefa (sem comprimentos, que dependem do tokenizador)."""
    raw = d.get('input_output') or {}
    # Só precisamos de fn_name. Entradas dos testes podem ter inteiros de milhares de dígitos.
    io_ = json.loads(raw, parse_int=str) if isinstance(raw, str) else raw
    fn = (io_ or {}).get('fn_name')
    if not fn:
        return None
    solutions = d.get('solutions') or []
    if isinstance(solutions, str):
        solutions = json.loads(solutions)
    prompt, formato, ind = apps_prompt(d['question'], d.get('starter_code') or '', fn)
    identifier = d['id'] if 'id' in d else d['problem_id']
    return dict(dataset='APPS', split=split, id=str(identifier), prompt=prompt, formato=formato,
                referencia=apps_referencia(solutions, fn, prompt, formato, ind),
                solucoes=[s for s in solutions if isinstance(s, str) and s.strip()],
                difficulty=d.get('difficulty') or '', target=fn)


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
    """HumanEval+, APPS e ClassEval (formato continuacao): lógica do notebook v4 (python_boundary).

    O prompt termina dentro da função (HumanEval+, APPS) ou do corpo da classe (ClassEval);
    ela acaba no próximo elemento no nível zero de indentação, fora de parênteses, que
    comece depois do prompt. No ClassEval isso é o fim da classe ("fim_da_classe").
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


def inicio_codigo(completion):
    """Posição onde o código começa: pula uma cerca ``` inicial e prosa antes do código."""
    m = _CERCA_INICIO.match(completion)
    base = m.end() if m else 0
    codigo = _INICIO_CODIGO.search(completion, base)
    return codigo.start() if codigo else base


def fronteira_bloco_alvo(completion, alvo):
    """APPS sem starter utilizável (alvo = fn_name). (Até o protocolo scripts-2, também o ClassEval.)

    O prompt não termina em código, então a completion é analisada sozinha. A mesma
    regra do v4 é aplicada a partir do bloco de nível zero que define `alvo`
    (`def alvo` ou `class alvo`, em qualquer profundidade): a fronteira é o próximo
    elemento de nível zero depois dele. Imports e funções auxiliares antes do alvo não
    encerram a geração. Uma cerca ``` que abre a resposta e prosa antes do código são
    ignoradas; uma cerca de fechamento depois do alvo também conta como fronteira.
    """
    base = inicio_codigo(completion)
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


def formato(task):
    return task.get('formato') or ('bloco' if task['dataset'] == 'APPS' else 'continuacao')


def fronteira(task, completion):
    if formato(task) == 'continuacao':
        return fronteira_continuacao(task['prompt'], completion)
    return fronteira_bloco_alvo(completion, task['target'])


def codigo_avaliavel(task, texto):
    """Programa a executar/compilar a partir de um trecho de resposta. No ClassEval, como
    no avaliador oficial (test_pipeline.py), os import_statement da tarefa vêm antes:
    alguns skeletons não trazem todos os imports que a classe usa (ex.: 39, 92)."""
    codigo = task['prompt'] + texto if formato(task) == 'continuacao' else texto[inicio_codigo(texto):]
    if task.get('imports'):
        codigo = '\n'.join(task['imports']) + '\n' + codigo
    return codigo


def compila(codigo):
    try:
        ast.parse(codigo)
        return True
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
        return False


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
    return dict(chars=pos, tokens=manter, motivo=motivo, texto=mantido, texto_inteira=texto,
                ast_ok=compila(codigo_avaliavel(task, mantido)),
                ast_ok_inteira=compila(codigo_avaliavel(task, texto)))
