# Demo clips: licence

The video files in this folder (`*.mp4`) and the stills in `web/public/thumbs/` are **not** covered by
this repository's Apache-2.0 licence. They are short, face-blurred cuts of third-party footage
published on Pexels under the [Pexels License](https://www.pexels.com/license/).

They are included only to demonstrate the tracker. **Do not reuse them as stock footage** from this
repository; if you want the footage, get it from the original Pexels page below, under Pexels' terms.

| File | Source (Pexels) |
|---|---|
| `subway_turnstile.mp4` | https://www.pexels.com/video/busy-subway-turnstile-in-urban-station-32511497/ |
| `mexico_city_entrance.mp4` | https://www.pexels.com/video/bustling-mexico-city-metro-entrance-30138564/ |
| `milan_turnstiles.mp4` | https://www.pexels.com/video/modern-milan-metro-station-turnstiles-31540854/ |
| `rabat_stadium.mp4` | https://www.pexels.com/video/fans-arriving-at-stadium-in-rabat-34452568/ |
| `man_entering_metro.mp4` | https://www.pexels.com/video/man-entering-a-metro-station-7251031/ |
| `shibuya_crossing.mp4` | https://www.pexels.com/video/a-crowded-city-street-with-many-people-walking-25947490/ |

Changes made here: cut to a short segment, scaled to 720p, faces blurred (soft ellipse about 1.2x each detected face, YuNet + the top of each person box; bodies left sharp; `python -m serve.blur --mode face`)
(`scripts/demo/make_demo_pack.sh`). The overlays drawn on top are the project's own output.
