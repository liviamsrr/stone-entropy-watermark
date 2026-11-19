"""Detecção: z-score por regra, AUROC e TPR com FPR fixo contra dois negativos.

    python src/analisar.py --config configs/humaneval.yaml          # ou --rodada resultados/<pasta>
    python src/analisar.py --rodada resultados/smoke_xxxx --bootstrap 200

Regras de contagem (z = (verdes - gamma*n) / sqrt(gamma*(1-gamma)*n); z = 0 se n = 0):
  todos       todos os tokens não especiais;
  stone       tokens não especiais e não sintáticos (o detector do STONE);
  stone_H>tau os do STONE com H > tau, para cada tau da rodada;
  natural     a regra do próprio braço positivo: stone_d* -> stone; hibrido_*_t<tau> -> stone_H>tau.
Positivos: braços stone_* e hibrido_*. Negativos: humano (referência por teacher forcing) e
sem_marca. Métricas para a resposta inteira e para a versão cortada no fim da função/classe,
em cada faixa de comprimento da referência (curta/média/longa, definidas antes da geração)
e em 'todas'. ICs: bootstrap percentil reamostrando tarefas (positivos e negativos da mesma
tarefa entram juntos). TPR@5% exige >= 20 negativos; TPR@1%, >= 100.

Saídas em <rodada>/analise/: respostas.csv, deteccao.csv, pareado_hibrido_stone.csv.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comum  # noqa: E402

VERSOES = ('inteira', 'cortada')
FPRS = {0.05: 20, 0.01: 100}    # alvo de FPR -> mínimo de negativos


# ----------------------------------------------------------- por resposta
def regras(taus):
    return ['todos', 'stone'] + [f'stone_H>{t}' for t in taus]


def regra_natural(braco):
    if braco.startswith('hibrido_'):
        return 'stone_H>' + braco.rsplit('_t', 1)[1]
    return 'stone'


def zscore(n, g, gamma):
    return (g - gamma * n) / np.sqrt(gamma * (1 - gamma) * n) if n else 0.0


def linhas_resposta(s, base, gamma, taus):
    ids = s['ids']
    H = np.asarray(s['H'], dtype=float)
    especial = np.asarray(s['special'], dtype=bool)
    contado = np.asarray(s['counted'], dtype=bool) & ~especial
    verde = np.asarray(s['green'], dtype=bool)
    out = []
    for versao in VERSOES:
        L = len(ids) if versao == 'inteira' else s['corte']['tokens']
        mascaras = {'todos': ~especial[:L], 'stone': contado[:L]}
        for t in taus:
            mascaras[f'stone_H>{t}'] = contado[:L] & (H[:L] > t)
        row = dict(base, versao=versao, tokens=int((~especial[:L]).sum()), ids=L)
        for nome, m in mascaras.items():
            n, g = int(m.sum()), int(verde[:L][m].sum())
            row[f'n_{nome}'], row[f'g_{nome}'], row[f'z_{nome}'] = n, g, zscore(n, g, gamma)
        out.append(row)
    return out


def tabela_respostas(run, taus, gamma):
    inv = pd.read_csv(run / 'inventario.csv', dtype={'id': str}).set_index('key')
    rows, falhas = [], 0
    for key, braco, amostra, path in comum.respostas_da_rodada(run):
        s = comum.read_gz(path)
        if s.get('status') != 'ok':
            falhas += 1
            continue
        meta = inv.loc[key]
        base = dict(dataset=meta['dataset'], split=meta['split'], id=meta['id'], key=key,
                    faixa_ref=meta['faixa_ref'], ref_tokens=meta['ref_tokens'], braco=braco,
                    amostra=-1 if amostra is None else amostra, stop=s.get('stop'),
                    corte=s['corte']['motivo'])
        rows.extend(linhas_resposta(s, base, gamma, taus))
    print(f'{len(rows) // 2} respostas lidas; {falhas} sem status ok (ignoradas).')
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- métricas
def auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    neg = np.sort(neg)
    menor = np.searchsorted(neg, pos, 'left')
    menor_igual = np.searchsorted(neg, pos, 'right')
    return float((menor + 0.5 * (menor_igual - menor)).sum() / (len(pos) * len(neg)))


def tpr_fpr(pos, neg, alvo):
    """Limiar = (f+1)-ésimo maior negativo, f = floor(alvo*m); detecta se z > limiar."""
    m = len(neg)
    if m < FPRS[alvo] or len(pos) == 0:
        return np.nan
    limiar = np.sort(neg)[::-1][int(np.floor(alvo * m))]
    return float((pos > limiar).mean())


def _matriz(df, coluna, chaves, amostras=None):
    """Tarefas x amostras (NaN onde falta) para a coluna pedida."""
    if df.empty:
        return np.full((len(chaves), len(amostras or [0])), np.nan)
    p = df.pivot_table(index='key', columns='amostra', values=coluna, aggfunc='first')
    if amostras is not None:
        p = p.reindex(columns=amostras)
    return p.reindex(chaves).to_numpy(dtype=float)


def _vals(M, idx):
    v = M[idx].ravel()
    return v[~np.isnan(v)]


def _ic(amostras):
    a = np.asarray(amostras, dtype=float)
    a = a[~np.isnan(a)]
    return (np.nan, np.nan) if len(a) < 2 else tuple(np.quantile(a, [0.025, 0.975]))


def _rng(seed, *partes):
    return np.random.default_rng([seed, int(comum.digest(list(map(str, partes)))[:8], 16)])


def deteccao(df, positivos, negativos, todas_regras, B, seed):
    out = []
    grupos = [(ds, sp, f, v) for (ds, sp, v), _ in df.groupby(['dataset', 'split', 'versao'], sort=False)
              for f in ['todas'] + sorted(df.faixa_ref.dropna().unique())]
    for ds, sp, faixa, versao in grupos:
        sub = df[(df.dataset == ds) & (df.split == sp) & (df.versao == versao)]
        if faixa != 'todas':
            sub = sub[sub.faixa_ref == faixa]
        for pos_b in positivos:
            P_df = sub[sub.braco == pos_b]
            for neg_b in negativos:
                N_df = sub[sub.braco == neg_b]
                chaves = sorted(set(P_df.key) & set(N_df.key))
                if not chaves:
                    continue
                for regra in todas_regras + ['natural']:
                    r = regra_natural(pos_b) if regra == 'natural' else regra
                    P, N = _matriz(P_df, f'z_{r}', chaves), _matriz(N_df, f'z_{r}', chaves)
                    tudo = np.arange(len(chaves))
                    pos, neg = _vals(P, tudo), _vals(N, tudo)
                    row = dict(dataset=ds, split=sp, faixa_ref=faixa, versao=versao, positivo=pos_b,
                               negativo=neg_b, regra=regra, regra_aplicada=r, tarefas=len(chaves),
                               n_pos=len(pos), n_neg=len(neg), z_medio_pos=pos.mean(), z_medio_neg=neg.mean(),
                               AUROC=auroc(pos, neg))
                    for a in FPRS:
                        row[f'TPR@{a:.0%}'] = tpr_fpr(pos, neg, a)
                    rng = _rng(seed, ds, sp, faixa, versao, pos_b, neg_b, regra)
                    boots = {k: [] for k in ['AUROC'] + [f'TPR@{a:.0%}' for a in FPRS]}
                    for _ in range(B):
                        idx = rng.integers(0, len(chaves), len(chaves))
                        bp, bn = _vals(P, idx), _vals(N, idx)
                        boots['AUROC'].append(auroc(bp, bn))
                        for a in FPRS:
                            boots[f'TPR@{a:.0%}'].append(tpr_fpr(bp, bn, a) if not np.isnan(row[f'TPR@{a:.0%}']) else np.nan)
                    for k, v in boots.items():
                        row[f'{k}_ic_low'], row[f'{k}_ic_high'] = _ic(v)
                    out.append(row)
    return pd.DataFrame(out)


def pareado(df, hibridos, negativos, todas_regras, B, seed):
    """Híbrido menos STONE com o mesmo delta, pareado por tarefa e amostra (mesma semente)."""
    out = []
    for (ds, sp, versao), sub_v in df.groupby(['dataset', 'split', 'versao'], sort=False):
        for faixa in ['todas'] + sorted(df.faixa_ref.dropna().unique()):
            sub = sub_v if faixa == 'todas' else sub_v[sub_v.faixa_ref == faixa]
            for hib in hibridos:
                stone = 'stone_d' + hib.split('_d', 1)[1].split('_t')[0]
                H_df, S_df = sub[sub.braco == hib], sub[sub.braco == stone]
                chaves = sorted(set(H_df.key) & set(S_df.key))
                if not chaves:
                    continue
                for regra in todas_regras + ['natural']:
                    rh = regra_natural(hib) if regra == 'natural' else regra
                    rs = 'stone' if regra == 'natural' else regra
                    am = sorted(set(H_df.amostra) | set(S_df.amostra))
                    Zh, Zs = _matriz(H_df, f'z_{rh}', chaves, am), _matriz(S_df, f'z_{rs}', chaves, am)
                    D = np.nanmean(Zh - Zs, axis=1)          # Δz médio por tarefa (amostras pareadas)
                    base = dict(dataset=ds, split=sp, faixa_ref=faixa, versao=versao, hibrido=hib, stone=stone,
                                regra=regra, regra_hibrido=rh, regra_stone=rs, tarefas=int((~np.isnan(D)).sum()))
                    rng = _rng(seed, 'par', ds, sp, faixa, versao, hib, regra)
                    idxs = [rng.integers(0, len(chaves), len(chaves)) for _ in range(B)]
                    dz = [np.nanmean(D[i]) for i in idxs]
                    out.append(dict(base, negativo='-', metrica='delta_z', valor=np.nanmean(D),
                                    ic_low=_ic(dz)[0], ic_high=_ic(dz)[1]))
                    for neg_b in negativos:
                        N_df = sub[sub.braco == neg_b]
                        if not set(chaves) & set(N_df.key):
                            continue
                        Nh, Ns = _matriz(N_df, f'z_{rh}', chaves), _matriz(N_df, f'z_{rs}', chaves)
                        tudo = np.arange(len(chaves))
                        metricas = {'delta_AUROC': lambda i: auroc(_vals(Zh, i), _vals(Nh, i)) - auroc(_vals(Zs, i), _vals(Ns, i))}
                        for a in FPRS:
                            metricas[f'delta_TPR@{a:.0%}'] = (lambda a: lambda i: tpr_fpr(_vals(Zh, i), _vals(Nh, i), a)
                                                              - tpr_fpr(_vals(Zs, i), _vals(Ns, i), a))(a)
                        for nome, f in metricas.items():
                            v = f(tudo)
                            ic = _ic([f(i) for i in idxs]) if not np.isnan(v) else (np.nan, np.nan)
                            out.append(dict(base, negativo=neg_b, metrica=nome, valor=v, ic_low=ic[0], ic_high=ic[1]))
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--rodada', type=Path)
    g.add_argument('--config', type=Path)
    ap.add_argument('--bootstrap', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--taus', type=float, nargs='*', default=None, help='padrão: taus de bracos.json')
    args = ap.parse_args(argv)
    warnings.filterwarnings('ignore', category=RuntimeWarning)   # médias de fatias vazias viram NaN
    run = comum.localizar_rodada(args.rodada, args.config)
    cfg = comum.read_gz(run / 'config.json.gz')
    info = json.loads((run / 'bracos.json').read_text(encoding='utf-8'))
    taus = sorted(args.taus if args.taus is not None else info['taus'])
    print(f'Rodada {run.name}: gamma={cfg["gamma"]}, taus={taus}, bootstrap={args.bootstrap}')

    df = tabela_respostas(run, taus, cfg['gamma'])
    if df.empty:
        raise SystemExit('Nenhuma resposta com status ok.')
    destino = run / 'analise'
    destino.mkdir(exist_ok=True)
    df.to_csv(destino / 'respostas.csv', index=False)

    bracos = sorted(df.braco.unique())
    positivos = [b for b in bracos if b.startswith(('stone_', 'hibrido_'))]
    negativos = [b for b in ('humano', 'sem_marca') if b in bracos]
    det = deteccao(df, positivos, negativos, regras(taus), args.bootstrap, args.seed)
    det.to_csv(destino / 'deteccao.csv', index=False)
    par = pareado(df, [b for b in positivos if b.startswith('hibrido_')], negativos, regras(taus),
                  args.bootstrap, args.seed)
    par.to_csv(destino / 'pareado_hibrido_stone.csv', index=False)

    resumo = det[(det.faixa_ref == 'todas') & (det.regra == 'natural')]
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(resumo[['dataset', 'split', 'versao', 'positivo', 'negativo', 'tarefas', 'AUROC',
                      'AUROC_ic_low', 'AUROC_ic_high', 'TPR@5%', 'TPR@1%']].round(3).to_string(index=False))
    print(f'CSVs em {destino}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
