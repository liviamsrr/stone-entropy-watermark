"""Baixa os conjuntos de dados para dados/, com os nomes que o notebook v5 usa.

Feito para a sessão de login (sem GPU): usa só a biblioteca padrão.
Arquivos que já existem não são baixados de novo.

    python src/baixar_dados.py                      # todos (~1,4 GB, quase tudo é APPS)
    python src/baixar_dados.py --conjuntos humaneval classeval
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comum import ARQUIVOS, CONJUNTOS, DADOS, fetch, sha256_arquivo  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--conjuntos', nargs='+', choices=sorted(CONJUNTOS), default=sorted(CONJUNTOS))
    ap.add_argument('--destino', type=Path, default=DADOS, help='padrão: dados/ na raiz do repositório')
    args = ap.parse_args()
    for nome in args.conjuntos:
        for fonte in CONJUNTOS[nome]['fontes']:
            existia = (args.destino / ARQUIVOS[fonte]).exists()
            path = fetch(fonte, args.destino)
            mb = path.stat().st_size / 2**20
            print(f'{fonte:12s} {"já existia" if existia else "baixado":10s} {mb:8.1f} MB  '
                  f'sha256={sha256_arquivo(path)[:16]}  {path}', flush=True)


if __name__ == '__main__':
    main()
