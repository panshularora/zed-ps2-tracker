#!/usr/bin/env bash
# Demo pack committed to the repo so a fresh clone plays out of the box (no `make clips blur`):
#   web/public/demo/<clip>.mp4   720p H.264 cut of the tracked range (from 0 s, so overlays align;
#                                offset stored in web/public/demo/index.json), face-blurred source
#   web/public/thumbs/<clip>.jpg 320x180 still for the Scenarios rail (cover crop to 16:9, no bars;
#   portrait clips keep the band a little below centre, where people walk)
# Source: the face-blurred clips from `make models clips blur` (data/demo/blurred, face-only oval
# blur). Never raw clips. A keyframe every 10 frames (-g 10), like the full blurred copies.
# The overlays (web/public/scenarios) are computed on the raw clips, not on these cuts.
set -euo pipefail
cd "$(dirname "$0")/../.."
SRC=${SRC:-data/demo/blurred}
mkdir -p web/public/demo web/public/thumbs
echo "{" > web/public/demo/index.json
first=1
for id in $(python3 -c "import json;print(' '.join(m['id'] for m in json.load(open('web/public/scenarios/index.json'))))"); do
  f=$(ls "$SRC"/"$id"_*.mp4 | head -1)
  dur=$(python3 -c "import json;print([m for m in json.load(open('web/public/scenarios/index.json')) if m['id']=='$id'][0]['covered_s'])")
  ffmpeg -loglevel error -y -i "$f" -t "$(python3 -c "print($dur+0.5)")" -an \
    -vf "scale='if(gt(iw,ih),-2,720)':'if(gt(iw,ih),720,-2)'" -c:v libx264 -preset slow -crf 30 \
    -g 10 -keyint_min 10 -sc_threshold 0 -pix_fmt yuv420p -movflags +faststart "web/public/demo/$id.mp4"
  ffmpeg -loglevel error -y -ss 2 -i "$f" -frames:v 1 -vf "scale=320:180:force_original_aspect_ratio=increase,crop=320:180:(iw-320)/2:(ih-180)*0.62" -q:v 6 "web/public/thumbs/$id.jpg"
  [ $first = 1 ] || echo "," >> web/public/demo/index.json
  first=0
  printf ' "%s": {"src": "/demo/%s.mp4", "offset_s": 0, "duration_s": %s}' "$id" "$id" "$dur" >> web/public/demo/index.json
done
echo "" >> web/public/demo/index.json
echo "}" >> web/public/demo/index.json
du -ch web/public/demo/*.mp4 web/public/thumbs/*.jpg | tail -1
