# TrackEval (vendored, unmodified)

* Upstream: https://github.com/JonathonLuiten/TrackEval
* Commit: `12c8791b303e0a0b50f753af204249e622d0281a` (29 Nov 2022, latest on master)
* Licence: MIT (`LICENSE`, copied verbatim)
* Contents: the `trackeval/` package only; no scripts, docs or tests.
* No source changes. NumPy ≥ 1.24 removed `np.float`/`np.int`/`np.bool`, which this commit
  still uses, so `eval/trackeval_runner.py::import_trackeval` re-adds them as aliases of the
  builtins before importing. That is what they were in older NumPy, so results are unchanged.
