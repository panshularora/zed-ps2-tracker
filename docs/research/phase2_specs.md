# Phase 2 specs from the room (Oct 4, 00:43 IST)

## Go/no-go test of the delayed decision on CPU (IDEA LAB + Research Lab)
- After delayed-decision matching lands, compare ByteTrack-style matching with matching plus the delayed decision, using motion and IoU only, at latency budgets K = 0, 3, 5, 8 frames.
- Inputs:
  (a) MOT17/MOT20 gt boxes, corrupted realistically: position jitter, drop probability rising as gt visibility falls, plus false positives.
  (b) ALSO the real public detections shipped in the labels zips (MOT17 FRCNN/SDP/DPM det.txt, MOT20 det.txt). These have genuine detector failures, merged and missed boxes, and partly answer Research Lab's caveat.
- Report:
  - ID switches inside the windows the tracker flags as ambiguous;
  - ALSO IDs, IDF1 and HOTA over all frames (Research Lab's requirement, because the windows-only count favours us);
  - the curve of ID switches against K, and p95 and max latency.
- Treat a positive result as a sanity check only, kept out of the results tables until the real DGX detections confirm it. If the result is flat, the pitch leads with the visibility estimate and gate events, and the delayed decision becomes an ablation. Be honest either way.

## Viewer output format (Web Experience Analyst)
One `<clip>.<tracker>.json.gz` per clip per tracker:
{ "clip","tracker","fps","frames","width","height", "gate":{"a":[x,y],"b":[x,y]},
  "tracks": { "<id>": [[frame,x,y,w,h,conf], ...] },   // pixels, frames 1-based
  "pending": [{ "f0","f1","candidates":[ids],"chosen":id }],
  "events": [ {"f","type":"id_switch","gt","from","to"}, {"f","type":"cross","id","dir":"in|out","correct":bool}, {"f","type":"tailgate","ids":[a,b]} ] }
- Events are precomputed with TrackEval-style matching plus gate_events.py. The viewer only draws and seeks.
- The viewer reads the gzip with DecompressionStream.
- It works tonight on gt tracks and public-det tracker outputs, with no video needed: draw on a blank canvas, or on frames if a clip exists.

## Viewer theme (3D Art Director), in viewer/src/theme.ts, shared with the deck
- ID colours by id % 20: #E69F00 #56B4E9 #009E73 #F0E442 #0072B2 #D55E00 #CC79A7 #8DD3C7 #FFFFB3 #BEBADA #FB8072 #80B1D3 #FDB462 #B3DE69 #FCCDE5 #BC80BD #CCEBC5 #FFED6F #1B9E77 #A6CEE3 (no red in the list).
- Held decision: #FFFFFF at 80% opacity, dashed setLineDash([6,4]). Error: #E5484D. Gate line: #F5A524, 2px.
- Boxes: 2px solid. Labels: 11px JetBrains Mono on a filled tab in the ID colour, text #0E1013.
- Background #0E1013; panels #161A1F with a 1px #262B33 border; text #E6E8EB; secondary text #8A929C. Inter for text, tabular-nums for counters, 4px radius.
- Motion: a decision fades from dashed to solid over 150ms; the gate pulses once over 300ms when someone crosses; seeking is instant; nothing else animates.
- The two videos are equal width, side by side, stacked vertically on narrow screens.
- The viewer is static: it opens from index.html with no server.
