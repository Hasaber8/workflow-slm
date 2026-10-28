PYTHON = ./venv/bin/python
PYTEST = ./venv/bin/pytest
UV ?= $(shell command -v uv 2>/dev/null || echo "$(HOME)/.local/bin/uv")
CODEX ?= $(shell command -v codex 2>/dev/null || echo "$(HOME)/.local/bin/codex")
EXTRAS ?= local
TRACE_ARGS ?=
LOGIN ?= 0
LOGIN_ARGS ?= --device-auth

.PHONY: setup login test smoke world data traces baselines sft train serve eval ablate report all

setup:
	@test -x $(PYTHON) || "$(UV)" venv venv --python 3.11
	"$(UV)" pip install --python $(PYTHON) -e ".[$(EXTRAS)]"
	@if [ "$(LOGIN)" = "1" ]; then "$(CODEX)" login $(LOGIN_ARGS); fi

login:
	"$(CODEX)" login $(LOGIN_ARGS)

test:
	$(PYTEST) -v tests/

smoke:
	$(PYTHON) -m exp.smoke

world:
	$(PYTEST) -v tests/test_world.py

data:
	$(PYTHON) -m exp.data.generate

traces:
	$(PYTHON) -m exp.teacher.run_traces $(TRACE_ARGS)

baselines:
	$(PYTHON) -m exp.baselines.run_baseline

sft:
	$(PYTHON) -m exp.train.build_sft

train:
	$(PYTHON) -m exp.train.train_lora

serve:
	bash src/exp/serve/serve.sh

eval:
	$(PYTHON) -m exp.eval.run_eval

ablate:
	$(PYTHON) -m exp.eval.ablate

report:
	$(PYTHON) -m exp.eval.report

all: test smoke
