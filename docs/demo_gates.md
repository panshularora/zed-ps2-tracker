# Demo gates: one rule for every clip

The counting line on each Pexels demo clip (Scenarios tab, and the clip's default gate in
`configs/demo/clips.yaml`, which is demo config, not research config) comes from **where people
walk**, using the same rule on every clip. No tracker's count is looked at when the line is chosen.

**Rule** (`scripts/demo/guide.py::motion_gate`):

1. Take every track of both trackers (pooled with equal weight, so neither is favoured) that lasts
   at least 1 s and whose foot point (box bottom-centre) moves at least 5 % of the frame diagonal.
2. Remove the camera pan first. Phone clips pan (`rabat_stadium` about 1,500 px, `milan_turnstiles`
   about 1,400 px), which would otherwise look like people walking sideways. The background shift
   per frame comes from phase correlation of consecutive frames on the unblurred clip
   (`scripts/demo/camera_motion.py`, stored as `camera` in `web/public/scenarios/<clip>.json`).
3. The dominant walking axis is the principal axis of the tracks' first-to-last foot displacements
   (opposite flows on the same axis add up).
4. The gate is the line **perpendicular** to that axis, through the median track mid-point along
   the axis, drawn across the whole frame.
5. "in" is the direction most of those tracks walk (the line's left-hand normal, the
   `eval/gate_events.py` convention).

A clip stays on the demo rail only if at least one tracker counts a crossing at its gate. Under
this rule no clip counts 0/0, so none was dropped. `tests/test_demo_gates.py` checks the rule on
synthetic walkers (including a panning camera), checks that swapping the two trackers gives the
same gate, and checks that every committed guide and `clips.yaml` gate equals the rule's output.

These clips have **no ground truth**, so the counts below are illustrative only and never results.
Counts are IN / OUT, with the first 1 s not counted (`SETTLE_S`) and 10 px hysteresis.

| Clip | Old gate | ByteTrack | Ours | New gate (x1, y1, x2, y2) | Walking axis | ByteTrack | Ours |
|---|---|---|---|---|---|---|---|
| man_entering_metro | h80 (horizontal, y 0.8) | 0 / 1 | 0 / 1 | 0.335, 1.0, 0.146, 0.0 | −18.6° (2 tracks) | 1 / 0 | 1 / 0 |
| subway_turnstile | h50 | 21 / 18 | 20 / 16 | 0.501, 1.0, 0.509, 0.0 | 0.8° (52 tracks) | 11 / 0 | 11 / 0 |
| milan_turnstiles | h65 | 2 / 3 | 1 / 2 | 0.292, 1.0, 0.235, 0.0 | −5.8° (16 tracks) | 5 / 0 | 5 / 0 |
| mexico_city_entrance | h50 | 1 / 1 | 1 / 1 | 0.117, 0.0, 0.601, 1.0 | 139.3° (2 tracks) | 1 / 0 | 1 / 0 |
| shibuya_crossing | v35 (vertical, x 0.35) | 7 / 51 | 5 / 56 | 0.49, 1.0, 0.45, 0.0 | −4.1° (166 tracks) | 40 / 3 | 44 / 3 |
| rabat_stadium (Fast Lane) | v35 | 14 / 0 | 14 / 0 | 0.0, 0.795, 1.0, 0.734 | 83.8° (23 tracks) | 11 / 4 | 10 / 3 |

The old guide gate was the fixed-fraction candidate line crossed most often by both trackers
summed, so it was chosen from counts; the new rule is not. The `clips.yaml` gates were all
horizontal (y 0.5 or 0.6) before. Angles are image degrees (0° = rightward, 90° = down the image).

**Limits.** `man_entering_metro` and `mexico_city_entrance` have only 2 moving tracks each, so
their axes rest on very little motion. On `mexico_city_entrance` the people on the stairs are
small and far away, and the line runs diagonally across the stairs. Both clips still count
someone, so they stay on the rail.
