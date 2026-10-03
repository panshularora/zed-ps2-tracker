# zed-ps2-tracker — developer entry points.
#   make setup   create .venv (Python 3.11) and install the package + dev tools
#   make check   lint + format check + mypy --strict tracker/ + tests   (what CI runs)
#   make data    download MOT17 labels + public detections only (~10 MB zip, no images)
#   make eval    run the configured tracker on MOT17 train public dets and score with TrackEval
#   make stress  IDF1 / IDsw binned by gt visibility and people-per-frame (needs `make eval` first)
#   make latency per-frame tracker latency p50 / p95 / max on this machine

PY      ?= .venv/bin/python
DATA    ?= data
OUT     ?= outputs
CONFIG  ?= configs/baselines/bytetrack.yaml
TRACKER ?= bytetrack_frcnn_public
SPLIT   ?= train
DETS    ?= FRCNN

.PHONY: setup check lint format typecheck test data eval stress latency clean

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

data: $(DATA)/MOT17/train

$(DATA)/MOT17/train:
	mkdir -p $(DATA)
	curl -fSL -o $(DATA)/MOT17Labels.zip https://motchallenge.net/data/MOT17Labels.zip
	unzip -q -o $(DATA)/MOT17Labels.zip -d $(DATA)/MOT17
	rm -f $(DATA)/MOT17Labels.zip

eval: $(DATA)/MOT17/train
	$(PY) -m eval.run_mot --config $(CONFIG) --mot-root $(DATA)/MOT17 --split $(SPLIT) \
		--dets $(DETS) --tracker-name $(TRACKER) --out $(OUT)
	$(PY) -m eval.trackeval_runner --gt-root $(DATA)/MOT17 --split $(SPLIT) --dets $(DETS) \
		--trackers-root $(OUT)/trackers --tracker-name $(TRACKER) --out $(OUT)/metrics

stress: $(DATA)/MOT17/train
	$(PY) -m eval.stress --gt-root $(DATA)/MOT17 --split $(SPLIT) --dets $(DETS) \
		--trackers-root $(OUT)/trackers --tracker-name $(TRACKER) --out $(OUT)/stress

latency: $(DATA)/MOT17/train
	$(PY) -m eval.latency --config $(CONFIG) --mot-root $(DATA)/MOT17 --split $(SPLIT) \
		--dets $(DETS) --out $(OUT)/latency

clean:
	rm -rf $(OUT) .mypy_cache .ruff_cache .pytest_cache
