# Gate events (our proposed metrics), MOT17 val_half, public FRCNN

Definitions are in `eval/gate_events.py`. They build on the line-crossing counting setup of Ma & Chan
(CVPR 2013) and add identity attribution and tailgating. The gates are `configs/gates/mot17_val_half.yaml`:
two lines per static-camera sequence (MOT17-02, -04, -09), placed at fixed fractions of the image
and frozen before the trackers were scored. Parameters: hysteresis 10 px, match window ±15 frames,
tailgate window 1 s. Tracker outputs are the `make ablate` runs.

Reproduce: `make gates` (after `make ablate`).

| Tracker | gt / trk crossings | count error % | event P / R % | attribution error % | tailgates gt / trk | tailgate P / R % |
|---|---|---|---|---|---|---|
| gt as tracker (check) | 25 / 25 | 0.0 | 100.0 / 100.0 | 0.0 | 9 / 9 | 100.0 / 100.0 |
| ByteTrackLike | 25 / 18 | 28.0 | 83.3 / 60.0 | 33.3 | 9 / 5 | 80.0 / 44.4 |
| OC-SORT | 25 / 15 | 40.0 | 93.3 / 56.0 | 14.3 | 9 / 5 | 80.0 / 44.4 |
| Deep OC-SORT | 25 / 15 | 40.0 | 93.3 / 56.0 | 14.3 | 9 / 5 | 80.0 / 44.4 |
| Hybrid-SORT | 25 / 16 | 36.0 | 93.8 / 60.0 | 20.0 | 9 / 5 | 80.0 / 44.4 |
| BoT-SORT | 25 / 17 | 32.0 | 88.2 / 60.0 | 33.3 | 9 / 5 | 80.0 / 44.4 |
| ByteTrackLike buffer90 | 25 / 19 | 24.0 | 78.9 / 60.0 | 26.7 | 9 / 6 | 66.7 / 44.4 |
| ByteTrackLike + visibility | 25 / 19 | 24.0 | 78.9 / 60.0 | 26.7 | 9 / 6 | 66.7 / 44.4 |
| ByteTrackLike + delayed K=5 | 25 / 25 | 0.0 | 60.0 / 60.0 | 46.7 | 9 / 5 | 60.0 / 33.3 |
| BoT-SORT + visibility | 25 / 18 | 28.0 | 83.3 / 60.0 | 26.7 | 9 / 6 | 66.7 / 44.4 |
| BoT-SORT buffer90 | 25 / 18 | 28.0 | 83.3 / 60.0 | 26.7 | 9 / 6 | 66.7 / 44.4 |


How to read this:
- **The sample is small: 25 gt crossings and 9 gt tailgates in total.** One event is 4 points of
  recall, so differences of one or two events between trackers are noise. These numbers show that
  the metric works end to end. They do not rank the trackers.
- `gt as tracker` scores 0 error, which checks the implementation.
- Every tracker finds about 60 % of the crossings: public FRCNN misses many of the partly visible
  people near the lines. Attribution error (a matched crossing given to the wrong identity under
  the IDF1 mapping) is 14–33 % for the baselines and 47 % with the delayed decision.
- The delayed row's 0 % count error is a coincidence: 25 vs 25 events, but only 60 % of them match.
  This is why count error alone is a poor metric and precision/recall/attribution are reported too.
