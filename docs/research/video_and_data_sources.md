# Extra Videos and Datasets for PS-2 (checked 4 Oct 2026)

## 1. Extra benchmark with a live test server: SOMPT22
- [SOMPT22](https://sompt22.github.io/) ([paper, arXiv 2208.02580](https://arxiv.org/abs/2208.02580)): surveillance-style pedestrian tracking from **static, elevated cameras on poles 6–8 m up**, close to a gate camera. 14 one-minute videos, 9 train with public GT, 5 test with images only. About 37 people per frame, between MOT17 and MOT20.
- **Its [test server](https://sompt22.github.io/challenge/) is open.** Results go in MOTChallenge `.txt` format, scored with TrackEval on HOTA, submitted through a GitHub issue. The limit is **1 submission per team per week**, and FPS has to be declared on stated hardware. The top 3 need a tech report.
- Since MOTChallenge's own test server is offline, this is our only route to a real hidden-test-set number. Plan for at most 2–3 submissions across the month.
- License: CC BY-NC-SA 4.0 (research/non-commercial, same caveat as MOT17/20).
- Download: the [Google Drive folder](https://drive.google.com/drive/folders/1Z_gnFmX-EKUe4yLBQPa2pxXkyqYbxkhX?usp=sharing) linked from the site. **Add it to tomorrow's DGX run.**

## 2. Densest-crowd stress test: CroHD / HT21 (optional)
- "Tracking Pedestrian Heads in Dense Crowd", CVPR 2021 ([paper](https://openaccess.thecvf.com/content/CVPR2021/html/Sundararaman_Tracking_Pedestrian_Heads_in_Dense_Crowd_CVPR_2021_paper.html)). Up to about 346 people per frame, average 178, filmed from an elevated view, including a **Shibuya train station** scene.
- It annotates **heads, not bodies**, so it's not a drop-in for our body tracker. Use it only as an extreme-density qualitative clip, or skip it.
- Archive: `https://motchallenge.net/data/HT21.zip` (about 4.4 GB, still downloadable).

## 3. Free demo clips of real gates (Pexels: free to use, commercial OK, no attribution required)
These are for the **demo, deck and explanation video**. They have no ground truth, so they're qualitative only, plus IDEA LAB's gate-event counts if we hand-label crossings.
- [Busy Subway Turnstile in Urban Station](https://www.pexels.com/video/busy-subway-turnstile-in-urban-station-32511497/), the closest match to Zed's scene
- [Modern Milan Metro Station Turnstiles](https://www.pexels.com/video/modern-milan-metro-station-turnstiles-31540854/)
- [Bustling Mexico City Metro Entrance](https://www.pexels.com/video/bustling-mexico-city-metro-entrance-30138564/)
- [Fans Arriving at Stadium in Rabat](https://www.pexels.com/video/fans-arriving-at-stadium-in-rabat-34452568/), a stadium-entrance crowd for the Fast Lane angle
- [Man Entering a Metro Station](https://www.pexels.com/video/man-entering-a-metro-station-7251031/), a single person at the gate, useful for the intro shot

Before using any of them, open each page, check the camera angle (we want high and static) and download the 1080p file. Don't present real people in a negative light, for example by labelling someone in a stock clip a "fare evader". Show tailgating alerts only on clips where it really happens, or on your own consented clip.

## 4. Why not X or Reddit videos
Clips posted on X or Reddit belong to whoever uploaded them, often show identifiable bystanders, and come with no reuse license. That's fine to watch for ideas, but **not something to put in a zip that goes to a company**. Use Pexels, the datasets above, or your own consented gate clip.

## 5. Practitioner lessons from r/computervision (supports our design)
- [Trouble with MOT in supermarkets](https://www.reddit.com/r/computervision/comments/1l7c5yn/trouble_with_mot_in_supermarkets_frequent_id/): BoT-SORT keeps switching IDs even with 2–5 people. The advice is that association fails for a few frames during partial occlusion, and an off-the-shelf OSNet re-ID has a domain gap at high camera angles. This is a real-world version of our problem, and it argues for (a) sweeping the matching thresholds and (b) checking that the re-ID embeddings actually separate people at gate angles.
- [Keeping IDs when people leave and re-enter](https://www.reddit.com/r/computervision/comments/1k98us6/how_can_i_maintain_consistent_person_ids_when/): practitioners say gallery-based re-ID "doesn't work that well in uncontrolled scenarios with a ton of people". That's why we keep re-association short-term and gated by visibility.

## 6. Related papers on counting at gates (cite in related work for IDEA LAB's gate-event metric)
- CountingMOT ([arXiv 2212.05861](https://arxiv.org/abs/2212.05861)) jointly learns counting, detection and re-ID for crowds.
- DR.VIC ([arXiv 2203.12335](https://arxiv.org/abs/2203.12335)) is about video individual counting. It argues that ID switches cause **over-counting**, which is exactly the point behind our gate-event metric.
- Phys-3D (SciTePress 2026, [PDF](https://www.scitepress.org/Papers/2026/144147/144147.pdf)) does crowd tracking and counting on railway platforms. It shows that ID fragmentation causes double counts at a counting band, and it uses in-band persistence for N frames. That's a good design reference for `eval/gate_events.py`.
