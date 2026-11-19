"""Corretude: pass@1 e taxa de ast.parse das respostas de uma rodada.

    python src/avaliar.py --config configs/humaneval.yaml          # ou --rodada resultados/<pasta>
    python src/avaliar.py --rodada resultados/apps_xxxx --apps-subconjunto 100 --apps-testes-max 20

- HumanEval+: evalplus (testes base + plus; passa se os dois passam).
- ClassEval: os testes unittest do próprio conjunto (coluna `test`), rodados junto da classe.
- APPS: testes call-based de input_output, comparados como no avaliador oficial do APPS
  (mesmo cabeçalho de imports); --apps-subconjunto N sorteia N tarefas.
Avalia a versão cortada no fim da função/classe (--versao inteira para a resposta toda).
O braço humano (referência) também é avaliado, como verificação do ambiente.

Nenhum código gerado roda neste processo: cada programa vai para src/execucao.py
(subprocesso, diretório temporário, ambiente limpo, limites de tempo e memória).
pass@1 por tarefa usa o estimador não enviesado 1 - C(n-c,1)/C(n,1) = c/n (n amostras, c corretas).
Saídas em <rodada>/avaliacao/: respostas.csv e pass1.csv.
"""
import argparse
import gzip
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from math import comb
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comum  # noqa: E402
import execucao  # noqa: E402

# Cabeçalho que o avaliador oficial do APPS acrescenta antes de cada solução.
APPS_CABECALHO = (
    'import sys\nimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\n'
    'import collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\n'
    'from functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\n'
    'import fractions\nfrom typing import List, Tuple\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\n')

APPS_EXECUTOR = r'''
import json, runpy, signal, sys
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)
cfg = json.load(open("testes.json", encoding="utf-8"))
res = {"passou": False, "testes": 0, "ok": 0, "motivo": ""}
def fim():
    sys.stdout.write("\n__STONE__" + json.dumps(res) + "\n"); sys.stdout.flush()
try:
    ns = runpy.run_path("solucao.py", run_name="solucao")
except BaseException as e:
    res["motivo"] = "erro_ao_carregar: " + type(e).__name__; fim(); sys.exit(0)
fn = cfg["fn_name"]
alvo = getattr(ns["Solution"](), fn, None) if "Solution" in ns else None
alvo = alvo or ns.get(fn)
if alvo is None:
    res["motivo"] = "funcao_ausente"; fim(); sys.exit(0)
class Tempo(Exception):
    pass
def alarme(*_):
    raise Tempo()
tem_alarme = hasattr(signal, "SIGALRM")
if tem_alarme:
    signal.signal(signal.SIGALRM, alarme)
def igual(saida, esperado):
    if isinstance(saida, tuple):
        saida = list(saida)
    if saida == esperado:
        return True
    if isinstance(esperado, list) and esperado and saida == esperado[0]:
        return True
    try:
        if isinstance(saida[0], tuple) and [list(x) for x in saida] == esperado[0]:
            return True
    except Exception:
        pass
    return False
for entrada, esperado in zip(cfg["inputs"], cfg["outputs"]):
    res["testes"] += 1
    try:
        if tem_alarme:
            signal.setitimer(signal.ITIMER_REAL, cfg["tempo_teste"])
        saida = alvo(*entrada)
        if tem_alarme:
            signal.setitimer(signal.ITIMER_REAL, 0)
    except Tempo:
        res["motivo"] = "tempo_no_teste_%d" % res["testes"]; break
    except BaseException as e:
        if tem_alarme:
            signal.setitimer(signal.ITIMER_REAL, 0)
        res["motivo"] = "excecao_no_teste_%d: %s" % (res["testes"], type(e).__name__); break
    if not igual(saida, esperado):
        res["motivo"] = "saida_errada_no_teste_%d" % res["testes"]; break
    res["ok"] += 1
else:
    res["passou"] = res["testes"] > 0
    res["motivo"] = "ok" if res["passou"] else "sem_testes"
fim()
'''


def pass_at_k(n, c, k=1):
    """Estimador não enviesado (Chen et al., 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def _resultado_marcado(r):
    for linha in reversed(r['stdout'].splitlines()):
        if linha.startswith('__STONE__'):
            return json.loads(linha[len('__STONE__'):])
    return None


# ------------------------------------------------------------- por conjunto
def avaliar_apps(itens, testes_por_id, args):
    def um(item):
        io_ = testes_por_id[item['id']]
        ins, outs = io_['inputs'], io_['outputs']
        if args.apps_testes_max:
            ins, outs = ins[:args.apps_testes_max], outs[:args.apps_testes_max]
        cfg = dict(fn_name=io_['fn_name'], inputs=ins, outputs=outs, tempo_teste=args.tempo_teste)
        r = execucao.executar({'solucao.py': APPS_CABECALHO + item['codigo'], 'executor.py': APPS_EXECUTOR,
                               'testes.json': json.dumps(cfg)}, ['executor.py'],
                              tempo_s=min(args.tempo_teste * max(len(ins), 1) + 10, args.tempo_max),
                              memoria_mb=args.memoria_mb, isolamento=args.isolamento)
        res = _resultado_marcado(r)
        if r['tempo_esgotado']:
            return False, 'tempo_total', r['isolamento']
        if res is None:
            return False, f"sem_resultado (rc={r['rc']})", r['isolamento']
        return bool(res['passou']), res['motivo'], r['isolamento']
    with ThreadPoolExecutor(args.processos) as ex:
        return list(ex.map(um, itens))


def avaliar_classeval(itens, tarefas, args):
    def um(item):
        prog = item['codigo'] + '\n\n\n' + tarefas[item['key']]['testes'] + '\n'
        r = execucao.executar({'prog.py': prog}, ['-m', 'unittest', 'discover', '-s', '.', '-p', 'prog.py'],
                              tempo_s=args.tempo_max, memoria_mb=args.memoria_mb, isolamento=args.isolamento)
        if r['tempo_esgotado']:
            return False, 'tempo_total', r['isolamento']
        if r['rc'] == 0:
            return True, 'ok', r['isolamento']
        ultima = (r['stderr'].strip().splitlines() or [''])[-1][:120]
        return False, f"rc={r['rc']}: {ultima}", r['isolamento']
    with ThreadPoolExecutor(args.processos) as ex:
        return list(ex.map(um, itens))


def avaliar_humaneval(itens, args):
    """Um único processo do evalplus para todas as respostas, com o conjunto reduzido às
    tarefas presentes (o evalplus exige amostras para todas as tarefas do arquivo)."""
    if not itens:
        return []
    import signal
    if not hasattr(signal, 'setitimer'):
        raise SystemExit('O evalplus limita o tempo com signal.setitimer (só Unix); fora do Linux/macOS ele '
                         'marca tudo como timeout. Avalie o HumanEval+ no cluster.')
    ids = {it['id'] for it in itens}
    with gzip.open(comum.fetch('HumanEval+'), 'rt', encoding='utf-8') as f:
        problemas = [l for l in f if json.loads(l)['task_id'] in ids]
    amostras = ''.join(json.dumps({'task_id': it['id'], 'solution': it['codigo']}) + '\n' for it in itens)
    mem = max(args.memoria_mb, args.evalplus_memoria_mb)
    r = execucao.executar(
        {'HumanEvalPlus.jsonl': ''.join(problemas), 'samples.jsonl': amostras},
        ['-m', 'evalplus.evaluate', '--dataset', 'humaneval', '--samples', 'samples.jsonl',
         '--parallel', str(args.processos), '--i_just_wanna_run'],
        tempo_s=args.evalplus_tempo, memoria_mb=mem, isolamento=args.isolamento,
        env_extra={'HUMANEVAL_OVERRIDE_PATH': '{tmp}/HumanEvalPlus.jsonl', 'XDG_CACHE_HOME': '{tmp}/cache',
                   'EVALPLUS_MAX_MEMORY_BYTES': str(mem * 2**20), 'TOKENIZERS_PARALLELISM': 'false'},
        ler=['samples_eval_results.json'])
    if 'samples_eval_results.json' not in r['saidas']:
        raise RuntimeError(f"evalplus falhou (rc={r['rc']}, tempo_esgotado={r['tempo_esgotado']}):\n"
                           f"{r['stderr'][-3000:]}")
    ev = json.loads(r['saidas']['samples_eval_results.json'])['eval']
    contador, out = {}, []
    for it in itens:   # o evalplus devolve, por tarefa, os resultados na ordem do arquivo de amostras
        i = contador.get(it['id'], 0)
        contador[it['id']] = i + 1
        res = ev[it['id']][i]
        assert res['solution'] == it['codigo'], 'evalplus: ordem dos resultados não confere'
        ok = res['base_status'] == 'pass' and res['plus_status'] == 'pass'
        out.append((ok, f"base={res['base_status']} plus={res['plus_status']}", r['isolamento']))
    return out


def testes_apps(ids_por_split):
    out = {}
    for split, ids in ids_por_split.items():
        path = comum.DADOS / comum.ARQUIVOS[f'APPS/{split}']
        if not path.exists():
            raise SystemExit(f'{path} não existe. Rode src/baixar_dados.py --conjuntos apps.')
        with path.open(encoding='utf-8') as f:
            for linha in f:
                d = json.loads(linha)
                pid = str(d['id'] if 'id' in d else d['problem_id'])
                if pid in ids:
                    out[pid] = json.loads(d['input_output'])
    return out


# --------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--rodada', type=Path)
    g.add_argument('--config', type=Path)
    ap.add_argument('--conjuntos', nargs='+', default=['HumanEval+', 'ClassEval', 'APPS'])
    ap.add_argument('--bracos', nargs='*', default=None, help='padrão: todos')
    ap.add_argument('--versao', choices=['cortada', 'inteira'], default='cortada')
    ap.add_argument('--apps-subconjunto', type=int, default=None, help='sortear N tarefas do APPS')
    ap.add_argument('--apps-testes-max', type=int, default=None, help='máximo de testes por tarefa do APPS')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tempo-teste', type=float, default=4.0, help='APPS: segundos por teste')
    ap.add_argument('--tempo-max', type=float, default=60.0, help='segundos por programa')
    ap.add_argument('--memoria-mb', type=int, default=2048)
    ap.add_argument('--evalplus-memoria-mb', type=int, default=8192)
    ap.add_argument('--evalplus-tempo', type=float, default=7200.0)
    ap.add_argument('--processos', type=int, default=4)
    ap.add_argument('--isolamento', choices=['auto', 'bwrap', 'nenhum'], default='auto',
                    help="auto: bwrap se funcionar, senão só subprocesso; bwrap: exige bwrap")
    args = ap.parse_args(argv)

    run = comum.localizar_rodada(args.rodada, args.config)
    _, tarefas = comum.tarefas_da_rodada(run)
    modo = execucao.modo_isolamento(args.isolamento)
    print(f'Rodada {run.name}; isolamento: {modo}')
    if modo != 'bwrap':
        print('AVISO: sem bwrap, o código roda em subprocesso com limites, mas pode ler caminhos '
              'absolutos fora do diretório temporário.')

    itens = []
    for key, braco, amostra, path in comum.respostas_da_rodada(run):
        t = tarefas.get(key)
        if t is None or t['dataset'] not in args.conjuntos or (args.bracos and braco not in args.bracos):
            continue
        s = comum.read_gz(path)
        if s.get('status') != 'ok':
            continue
        # cortada: até o fim da função/classe; inteira: até o primeiro token especial (EOS etc.)
        texto = s['corte']['texto'] if args.versao == 'cortada' else s['corte']['texto_inteira']
        codigo = comum.codigo_avaliavel(t, texto)
        itens.append(dict(dataset=t['dataset'], split=t['split'], id=t['id'], key=key, braco=braco,
                          amostra=-1 if amostra is None else amostra, codigo=codigo,
                          ast_ok=comum.compila(codigo), ast_ok_cortada=s['corte']['ast_ok'],
                          ast_ok_inteira=s['corte']['ast_ok_inteira']))

    apps = [it for it in itens if it['dataset'] == 'APPS']
    if args.apps_subconjunto and apps:
        chaves = sorted({it['key'] for it in apps})
        escolha = set(random.Random(args.seed).sample(chaves, min(args.apps_subconjunto, len(chaves))))
        itens = [it for it in itens if it['dataset'] != 'APPS' or it['key'] in escolha]
        apps = [it for it in apps if it['key'] in escolha]
    print(f'{len(itens)} respostas a avaliar: '
          f'{pd.Series([it["dataset"] for it in itens]).value_counts().to_dict()}')

    resultados = {}
    he = [it for it in itens if it['dataset'] == 'HumanEval+']
    for it, r in zip(he, avaliar_humaneval(he, args)):
        resultados[id(it)] = r
    ce = [it for it in itens if it['dataset'] == 'ClassEval']
    for it, r in zip(ce, avaliar_classeval(ce, tarefas, args)):
        resultados[id(it)] = r
    if apps:
        ids = {}
        for it in apps:
            ids.setdefault(it['split'], set()).add(it['id'])
        for it, r in zip(apps, avaliar_apps(apps, testes_apps(ids), args)):
            resultados[id(it)] = r

    linhas = []
    for it in itens:
        ok, motivo, iso = resultados[id(it)]
        linhas.append({k: v for k, v in it.items() if k != 'codigo'} | dict(versao=args.versao, passou=ok,
                                                                           motivo=motivo, isolamento=iso))
    df = pd.DataFrame(linhas)
    destino = run / 'avaliacao'
    destino.mkdir(exist_ok=True)
    df.to_csv(destino / 'respostas.csv', index=False)
    if df.empty:
        print('Nada avaliado.')
        return 0
    inv = pd.read_csv(run / 'inventario.csv', dtype={'id': str})[['key', 'faixa_ref']]
    por_tarefa = (df.groupby(['dataset', 'split', 'key', 'braco'])
                  .agg(n=('passou', 'size'), c=('passou', 'sum'), ast=('ast_ok', 'mean')).reset_index()
                  .merge(inv, on='key', how='left'))
    por_tarefa['pass@1'] = [pass_at_k(n, c, 1) for n, c in zip(por_tarefa.n, por_tarefa.c)]
    resumo = []
    for faixa in ['todas'] + sorted(por_tarefa.faixa_ref.dropna().unique()):
        sub = por_tarefa if faixa == 'todas' else por_tarefa[por_tarefa.faixa_ref == faixa]
        for (ds, sp, br), g_ in sub.groupby(['dataset', 'split', 'braco']):
            resp = df[df.key.isin(g_.key) & (df.braco == br)]
            resumo.append(dict(dataset=ds, split=sp, faixa_ref=faixa, braco=br, tarefas=len(g_),
                               respostas=len(resp), pass_at_1=g_['pass@1'].mean(),
                               ast_parse=resp.ast_ok.mean(), ast_parse_cortada=resp.ast_ok_cortada.mean(),
                               ast_parse_inteira=resp.ast_ok_inteira.mean()))
    resumo = pd.DataFrame(resumo)
    resumo.to_csv(destino / 'pass1.csv', index=False)
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(resumo[resumo.faixa_ref == 'todas'].round(3).to_string(index=False))
    print(f'CSVs em {destino}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
