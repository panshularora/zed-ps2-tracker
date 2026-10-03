# Mentor Questions: After Phase 1 (PS-2, Zed Digital)
*Ask these once the baseline, the scoring pipeline and the stress bins are working. Lead with a 60-second status: what's built, first baseline numbers, and the planned headline idea.*

## A. Submission and judging (ask first, these change everything)
1. What exactly goes in the final submission: code repo, written report, video, live demo, deck? Are there page or length limits?
2. How are the top 5 chosen? Is there a rubric, and roughly how is it weighted between metrics, novelty, code quality, presentation and business fit?
3. Is there a private or hidden test set or video you'll run our tracker on? If yes, in what format should the tracker take input and write output (MOTChallenge `.txt` format)?
4. The MOTChallenge evaluation server is now offline, so MOT17 and MOT20 test-set scores can't be computed anymore. Is it fine to report results on held-out halves of the training sets plus DanceTrack val, or do you have a preferred benchmark split?
5. Is there a final presentation round or interview, and who is on the panel: research, engineering or product people?
6. What is the exact deadline, including the time zone, and can we resubmit before it?

## B. Problem scope and constraints
7. What does "real-time" mean for you: a target FPS, a latency budget per frame, and on what hardware (data-centre GPU, Jetson-class edge device, CPU)?
8. Is a bounded decision delay acceptable, for example committing an identity 3–5 frames (about 100–170 ms) late when two people are ambiguous, as long as everything stays online and causal?
9. Is the camera at your gates typically fixed and overhead or angled? Roughly what height and field of view, and how many people per frame at peak?
10. Single camera only, or should we design for hand-off between several cameras at one gate?
11. Are pretrained detectors and re-ID models allowed (YOLOX trained on CrowdHuman/MOT, OSNet), or must everything be trained from scratch?
12. Can you share any sample footage from a real deployment, even a short anonymised clip, to sanity-check our method?

## C. Technical direction (get feedback on our idea)
13. Our headline is a bounded-latency, online form of multiple hypothesis tracking that only holds a decision open between near-equal candidates. Does that match what you see failing in practice, or are your main failures elsewhere (missed detections, long disappearances, camera motion)?
14. We estimate each person's visibility from box overlap and depth order and use it to gate appearance updates. Is that cue reliable at your camera angles?
15. Which matters more to you if we have to trade: fewer ID switches (IDF1/AssA) or better detection coverage (MOTA/DetA)?
16. Are there baselines you'd specifically expect us to beat (ByteTrack, OC-SORT, BoT-SORT, Deep OC-SORT, Hybrid-SORT, something internal)?

## D. Zed ecosystem and integration
17. In ZIG, where would a tracker sit: on the validator/edge device, a local gate server, or the cloud? What does the downstream system expect from us (per-frame tracks, gate-crossing events, alerts)?
18. Which gate failure costs Zed the most today: tailgating, miscounting, or charging the wrong person? (This decides which number leads the pitch.)
19. We plan to score gate events too: crossing count error, crossings credited to the wrong person, and tailgating precision/recall. Are those the numbers your Fast Lane and fare-evasion products actually care about? Any others?
20. What are your privacy requirements? We keep only short-lived body-appearance embeddings in memory, with no faces and no stored identities. Is that compatible with your deployments?
21. Licensing: we're defaulting to Apache-2.0 components (YOLOX) and avoiding AGPL ones. Any licenses you require or forbid?

## E. Wrap-up
22. Based on what we've shown, what is the single biggest weakness you'd fix first?
23. What would make a submission stand out to your team beyond the metrics?
24. Can we check in again after phase 2 (the ablations) for a second review?
