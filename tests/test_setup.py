from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")
@pytest.mark.parametrize("existing_venv", [False, True])
@pytest.mark.parametrize("extras", [None, "", "local,serve"])
def test_setup_installs_selected_dependencies(tmp_path, existing_venv, extras):
    makefile = Path(__file__).resolve().parents[1] / "Makefile"
    uv = tmp_path / "fake uv"
    uv.write_text('#!/bin/sh\nprintf \'%s\\n\' "$*" >> uv-calls.log\n')
    uv.chmod(0o755)

    if existing_venv:
        python = tmp_path / "venv" / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)

    command = ["make", "-f", str(makefile), "setup", f"UV={uv}"]
    if extras is not None:
        command.append(f"EXTRAS={extras}")
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)

    calls = (tmp_path / "uv-calls.log").read_text().splitlines()
    selected_extras = "local" if extras is None else extras
    expected = [f"pip install --python ./venv/bin/python -e .[{selected_extras}]"]
    if not existing_venv:
        expected.insert(0, "venv venv --python 3.11")
    assert calls == expected


@pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")
@pytest.mark.parametrize("target", ["setup", "login"])
@pytest.mark.parametrize("login_args", [None, ""])
def test_optional_codex_login(tmp_path, target, login_args):
    makefile = Path(__file__).resolve().parents[1] / "Makefile"
    uv = tmp_path / "fake uv"
    uv.write_text("#!/bin/sh\nexit 0\n")
    uv.chmod(0o755)
    codex = tmp_path / "fake codex"
    codex.write_text('#!/bin/sh\nprintf \'%s\\n\' "$*" >> codex-calls.log\n')
    codex.chmod(0o755)
    command = ["make", "-f", str(makefile), target, f"UV={uv}", f"CODEX={codex}", "LOGIN=1"]
    if login_args is not None:
        command.append(f"LOGIN_ARGS={login_args}")
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)
    expected = "login --device-auth" if login_args is None else "login"
    assert (tmp_path / "codex-calls.log").read_text().strip() == expected


@pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")
def test_setup_reports_login_failure(tmp_path):
    makefile = Path(__file__).resolve().parents[1] / "Makefile"
    executable = tmp_path / "fake-cli"
    executable.write_text('#!/bin/sh\nif [ "$1" = "login" ]; then exit 7; fi\n')
    executable.chmod(0o755)
    result = subprocess.run(
        ["make", "-f", str(makefile), "setup", f"UV={executable}", f"CODEX={executable}", "LOGIN=1"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode != 0
