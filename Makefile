PYTHON = ./venv/bin/python
PYTEST = ./venv/bin/pytest

.PHONY: setup test smoke world data traces baselines sft train serve eval ablate report all

setup:
	uv venv venv --python 3.11
	uv pip install --python ./venv/bin/python -e .

test:
	$(PYTEST) -v tests/

smoke:
	$(PYTHON) -m exp.smoke

world:
	$(PYTEST) -v tests/test_world.py

data:
	$(PYTHON) -m exp.data.generate

traces:
	$(PYTHON) -m exp.teacher.run_traces

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
