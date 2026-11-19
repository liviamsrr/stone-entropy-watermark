"""Execução de código gerado fora do processo principal (só biblioteca padrão).

Cada execução:
- roda em subprocesso (`python -I`), num diretório temporário fora do projeto,
  apagado no fim;
- recebe um ambiente mínimo (PATH, HOME/TMPDIR no temporário, locale), sem as
  variáveis do processo pai, portanto sem nada vindo do .env;
- no Linux, tem limites de memória (RLIMIT_AS), CPU, tamanho de arquivo e core dump,
  e roda numa sessão própria, encerrada inteira ao estourar o tempo;
- com bubblewrap (bwrap) disponível e funcional, a pasta do projeto e a home ficam
  invisíveis (tmpfs por cima), a rede é cortada e o resto do sistema é só leitura.

Sem bwrap, o código continua num subprocesso com os limites acima, mas consegue ler
caminhos absolutos do sistema de arquivos: não é uma fronteira de segurança contra
código malicioso. O modo de isolamento usado é devolvido em cada resultado.
"""
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
LINUX = sys.platform.startswith('linux')


def _limites(memoria_mb, cpu_s, arquivo_mb):
    def aplicar():
        import resource
        mem = int(memoria_mb) * 2**20
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        resource.setrlimit(resource.RLIMIT_CPU, (int(cpu_s), int(cpu_s) + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (int(arquivo_mb) * 2**20,) * 2)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return aplicar


def _ambiente(tmp):
    env = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': tmp, 'TMPDIR': tmp, 'TEMP': tmp, 'TMP': tmp,
           'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8', 'PYTHONHASHSEED': '0',
           'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONIOENCODING': 'utf-8',
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    if os.name == 'nt':   # o Python do Windows não inicia sem estas
        for k in ('SYSTEMROOT', 'SYSTEMDRIVE', 'WINDIR', 'COMSPEC', 'PATHEXT'):
            if k in os.environ:
                env[k] = os.environ[k]
        env['PATH'] = os.environ.get('SYSTEMROOT', r'C:\Windows') + r'\System32'
    return env


def _ocultar():
    """Pastas que o código executado não deve ver."""
    home = Path.home()
    return [RAIZ] + ([home] if home != Path('/') else [])


def _bwrap_args(tmp):
    args = ['bwrap', '--ro-bind', '/', '/', '--dev', '/dev', '--proc', '/proc',
            '--tmpfs', '/tmp', '--unshare-all', '--die-with-parent', '--new-session']
    for p in sorted(set(_ocultar()), key=lambda p: len(p.parts)):
        if p.exists():
            args += ['--tmpfs', str(p)]
    # O interpretador (ex.: .venv dentro do projeto) volta como só leitura.
    for p in {Path(sys.prefix).resolve(), Path(sys.executable).resolve().parent}:
        if any(p.is_relative_to(o) for o in _ocultar()) and p.exists():
            args += ['--ro-bind', str(p), str(p)]
    args += ['--bind', tmp, tmp, '--chdir', tmp, '--']
    return args


@lru_cache(maxsize=1)
def bwrap_funciona():
    if not LINUX or shutil.which('bwrap') is None:
        return False
    tmp = tempfile.mkdtemp(prefix='stone_exec_teste_')
    try:
        r = subprocess.run(_bwrap_args(tmp) + [sys.executable, '-I', '-c', 'print(1)'],
                           capture_output=True, timeout=30, env=_ambiente(tmp))
        return r.returncode == 0 and r.stdout.strip() == b'1'
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def modo_isolamento(pedido='auto'):
    if pedido == 'nenhum':
        return 'subprocesso'
    if bwrap_funciona():
        return 'bwrap'
    if pedido == 'bwrap':
        raise RuntimeError('Isolamento bwrap pedido, mas o bwrap não está disponível ou não funciona aqui.')
    return 'subprocesso'


def executar(arquivos, argv, tempo_s=10.0, memoria_mb=2048, arquivo_mb=64, isolamento='auto',
             env_extra=None, ler=(), max_saida=20000):
    """Grava `arquivos` ({nome: texto}) num diretório temporário e roda
    `python -I <argv...>` lá dentro. Devolve dict com rc, stdout, stderr, tempo_esgotado,
    isolamento e `saidas` ({nome: texto} dos arquivos em `ler`, lidos antes de apagar o
    diretório). Nunca importa nem executa o código no processo atual."""
    modo = modo_isolamento(isolamento)
    tmp = tempfile.mkdtemp(prefix='stone_exec_')
    try:
        for nome, texto in arquivos.items():
            destino = Path(tmp) / nome
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_text(texto, encoding='utf-8')
        env = _ambiente(tmp)
        if env_extra:
            env.update({k: v.replace('{tmp}', tmp) for k, v in env_extra.items()})
        cmd = [sys.executable, '-I', *argv]
        if modo == 'bwrap':
            cmd = _bwrap_args(tmp) + cmd
        kw = dict(cwd=tmp, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if LINUX:
            kw.update(start_new_session=True, preexec_fn=_limites(memoria_mb, tempo_s + 1, arquivo_mb))
        proc = subprocess.Popen(cmd, **kw)
        esgotado = False
        try:
            out, err = proc.communicate(timeout=tempo_s)
        except subprocess.TimeoutExpired:
            esgotado = True
            if LINUX:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                proc.kill()
            out, err = proc.communicate()
        dec = lambda b: b.decode('utf-8', 'replace')[-max_saida:]  # noqa: E731
        saidas = {}
        for nome in ler:
            alvo = Path(tmp) / nome
            if alvo.is_file() and not alvo.is_symlink():
                saidas[nome] = alvo.read_text(encoding='utf-8', errors='replace')
        return dict(rc=proc.returncode, stdout=dec(out), stderr=dec(err), tempo_esgotado=esgotado,
                    isolamento=modo, limites=LINUX, saidas=saidas)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
