# zed-ps2-tracker: developer entry points.
#   make setup    create .venv (Python 3.11) and install the package + dev tools
#   make check    lint + format check + mypy (strict on tracker/) + tests   (what CI runs)
#   make data     download MOT17 + MOT20 labels and public detections only (no images, ~30 MB)
#   make eval     track DATASET/SPLIT with CONFIG on DETS, score with TrackEval, stress bins
#   make stress   re-run the stress bins on existing outputs of `make eval`
#   make latency  single-threaded tracker latency p50 / p95 / p99 / max on this CPU
#   make oracle   visibility-oracle ceiling (gt boxes with vis > 0 submitted as a tracker)
#   make ablate   ablation table + stress plots + latency for configs/ablations/manifest.yaml
#   make gates    gate-event metrics (crossings, attribution, tailgating) on the ablate outputs
#   make viewer   export demo clips + build the static viewer (viewer/dist/index.html)
#   make ingest   BRING_BACK=<tar|dir>: verify a Stage A bring-back (sha256) into data/stagea
#   make phase3   every post-GPU experiment from the ingested caches -> docs/results/phase3/
#   make phase3-selftest  ingest + phase3 on a synthetic Stage A bundle (no GPU data needed)
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

.PHONY: setup check lint format typecheck test data eval stress latency oracle smoke ablate gates viewer ingest phase3 phase3-selftest clean

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

# `make ablate`: every entry of configs/ablations/manifest.yaml (parallel tracking, then sequential
# single-threaded latency) -> docs/results/ablation/{ablation.md,csv,json, stress_*.png}
JOBS ?= 6
ablate: $(DATA)/MOT17/train
	$(PY) -m eval.ablate --manifest configs/ablations/manifest.yaml --out outputs/ablate \
		--report docs/results/ablation --jobs $(JOBS)

# `make gates`: gate-event metrics (eval/gate_events.py) on the `make ablate` outputs
gates:
	$(PY) -m eval.gate_events --gates configs/gates/mot17_val_half.yaml \
		--gt-dir outputs/ablate/bytetrack/gt --include-gt \
		--runs $(foreach n,bytetrack ocsort deep_ocsort hybrid_sort botsort buffer90 ours_vis ours_delayed_k5 ours_vis_botsort botsort_buffer90,outputs/ablate/$(n)/trackers/$(n)) \
		--out docs/results/gate_events

# `make viewer`: export demo clips and build the static viewer (viewer/dist/index.html)
viewer:
	$(PY) -m eval.export_viewer --seq MOT17-04-FRCNN MOT17-09-FRCNN \
		--gates configs/gates/mot17_val_half.yaml --include-gt --out viewer/data \
		--tracker bytetrack=configs/baselines/bytetrack.yaml \
		--tracker botsort=configs/baselines/botsort.yaml \
		--tracker delayed_k5=configs/ours/delayed.yaml
	cd viewer && npm ci && npm run build

# `make ingest BRING_BACK=<tar | folder of tars | bring_back dir>`: sha256-verify against the DGX
# SHA256SUMS, validate caches, place them under data/stagea/ and update data/stagea/index.json.
# Re-run per tar (MOT17 + MOT20 first). DELETE=1 removes each tar after it is verified.
STAGEA ?= data/stagea
ingest:
	@test -n "$(BRING_BACK)" || { echo "usage: make ingest BRING_BACK=<path> [DELETE=1]"; exit 2; }
	$(PY) -m eval.ingest --bring-back "$(BRING_BACK)" --dest $(STAGEA) --data $(DATA) \
		$(if $(filter 1,$(DELETE)),--delete-source,)

# `make phase3`: configs/phase3/plan.yaml on everything ingested so far (re-run after each ingest)
phase3:
	$(PY) -m eval.phase3 --plan configs/phase3/plan.yaml --index $(STAGEA)/index.json \
		--out outputs/phase3 --report docs/results/phase3 --jobs $(JOBS)

# `make phase3-selftest`: synthetic bundle (real Stage A writer, bringback, pack) -> ingest ->
# phase3, all in a temp dir that is deleted afterwards
phase3-selftest:
	@t=$$(mktemp -d) && trap 'rm -rf "$$t"' EXIT && \
	$(PY) -m eval.synth_stagea --root $$t/zed && \
	$(PY) -m eval.ingest --bring-back $$t/zed/bring_back_tars --dest $$t/stagea --data $$t/none && \
	$(PY) -m eval.phase3 --index $$t/stagea/index.json --out $$t/out --report $$t/report --jobs $(JOBS) && \
	cat $$t/report/README.md && echo "PHASE3 SELFTEST OK"
