# zed-ps2-tracker: developer entry points.
#   make setup    create .venv (Python 3.11) and install the package + dev tools
#   make check    lint + format check + mypy (strict on tracker/) + tests   (what CI runs)
#   make data     download MOT17 + MOT20 labels and public detections only (no images, ~30 MB)
#   make eval     track DATASET/SPLIT with CONFIG on DETS, score with TrackEval, stress bins
#   make stress   re-run the stress bins on existing outputs of `make eval`
#   make latency  single-threaded tracker latency p50 / p95 / p99 / max on this CPU
#   make oracle   visibility-oracle ceiling (gt boxes with vis > 0 submitted as a tracker)
#   make smoke    the phase-1 pipeline check: MOT17 train (full) and val-half, public FRCNN dets
#
# Defaults reproduce the README smoke numbers; override e.g.
#   make eval DATASET=MOT20 CONFIG=configs/baselines/bytetrack_mot20.yaml
#   make eval DETS=/path/to/cache/MOT17/ablation/train NAME=bytetrack_yolox

PY      ?= .venv/bin/python
DATA    ?= data
DATASET ?= MOT17
SPLIT   ?= val_half
DETS    ?= public
CONFIG  ?= configs/baselines/bytetrack.yaml
NAME    ?= bytetrack_public
OUT     ?= outputs/$(DATASET)_$(SPLIT)
BENCH    = $(if $(filter MOT20,$(DATASET)),MOT20,MOT17)

.PHONY: setup check lint format typecheck test data eval stress latency oracle smoke clean

setup:
	uv venv -p 3.11 .venv
	uv pip install -p .venv -e ".[dev]"

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

typecheck:
	$(PY) -m mypy --strict tracker
	$(PY) -m mypy eval models

test:
	$(PY) -m pytest -q

check: lint typecheck test

data: $(DATA)/MOT17/train $(DATA)/MOT20/train

$(DATA)/MOT17/train:
	mkdir -p $(DATA)
	curl -fSL -o $(DATA)/MOT17Labels.zip https://motchallenge.net/data/MOT17Labels.zip
	unzip -q -o $(DATA)/MOT17Labels.zip -d $(DATA)/MOT17
	rm -f $(DATA)/MOT17Labels.zip

$(DATA)/MOT20/train:
	mkdir -p $(DATA)
	curl -fSL -o $(DATA)/MOT20Labels.zip https://motchallenge.net/data/MOT20Labels.zip
	unzip -q -o $(DATA)/MOT20Labels.zip -d $(DATA)/MOT20
	rm -f $(DATA)/MOT20Labels.zip

eval: $(DATA)/$(DATASET)/train
	$(PY) -m eval.run --dataset $(DATASET) --root $(DATA)/$(DATASET) --split $(SPLIT) \
		--dets $(DETS) --config $(CONFIG) --name $(NAME) --out $(OUT)

stress:
	$(PY) -m eval.stress --gt $(OUT)/gt --trackers $(OUT)/trackers --tracker-name $(NAME) \
		--benchmark $(BENCH) --out $(OUT)/stress/$(NAME)

latency: $(DATA)/$(DATASET)/train
	OMP_NUM_THREADS=1 $(PY) -m eval.latency --dataset $(DATASET) --root $(DATA)/$(DATASET) \
		--split $(SPLIT) --dets $(DETS) --config $(CONFIG) --out $(OUT)/latency/$(NAME).json

oracle: $(DATA)/$(DATASET)/train
	$(PY) -m eval.oracle --dataset $(DATASET) --root $(DATA)/$(DATASET) --split $(SPLIT) \
		--tau 0 0.1 0.25 --out $(OUT)/oracle

smoke: $(DATA)/MOT17/train
	$(MAKE) eval SPLIT=full
	$(MAKE) eval SPLIT=val_half
	$(MAKE) latency SPLIT=full

clean:
	rm -rf outputs .mypy_cache .ruff_cache .pytest_cache
