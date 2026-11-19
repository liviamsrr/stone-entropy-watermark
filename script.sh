#!/bin/bash
# Comandos do job, em bash. Rode com:
#   sauron submit -c job.yaml script.sh
# CONFIG e ARGS vêm do env: do job.yaml.

set -e

# segredos (ex.: HF_TOKEN) ficam num .env na pasta do projeto, nunca no job.yaml
source .env 2>/dev/null || true

# dependências num venv na pasta do projeto, sobre os pacotes da imagem
export PYTHONNOUSERSITE=1
python -m venv --without-pip --system-site-packages .venv
. .venv/bin/activate
python -m pip install -r requirements.txt

# o torch tem de continuar sendo o da imagem (não reinstalado no .venv)
python - <<'EOF'
import sys, torch
if '.venv' in torch.__file__:
    sys.exit(f'torch foi instalado no .venv ({torch.__file__}); confira o requirements.txt')
print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'GPU', torch.cuda.is_available())
EOF

# HF_HOME já aponta para o cache da máquina; não alterar.
python src/gerar.py --config "${CONFIG:-configs/smoke.yaml}" ${ARGS}
