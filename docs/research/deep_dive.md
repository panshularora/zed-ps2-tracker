# PS-2 Deep Dive: Identity-Preserving Multi-Object Tracking through Dense Occlusion
*Zed Digital hackathon · prepared 4 Oct 2026*

## 1. The problem in one paragraph
Track every person walking through a **crowded gate**, in real time, online (no peeking at future frames), and keep **one stable ID per person** even when people block each other or vanish for a moment. When someone re-appears, they must get their **old ID back**, not a new one, and in tight groups IDs must **not jump onto a neighbour**. Performance must **degrade gracefully** as the crowd gets denser.

## 2. What they will actually judge (read the PS literally)
| PS says | What it really means for scoring |
|---|---|
| "Data association (Hungarian/graph matching)" | Show a principled cost matrix + assignment, not a black box |
| "Motion modelling (Kalman) fused with appearance" | A motion model + re-ID embeddings, combined with a *justified* fusion rule |
| "Track lifecycle management" | Explicit states: tentative, confirmed, occluded/lost, deleted, with tuned thresholds |
| "Re-association after loss" | A memory of lost tracks that can be recovered seconds later |
| "Robustness of association cost under partial observation" | **The core research hook.** Cost must not trust appearance/IoU when a person is half-hidden |
| "Online, bounded latency" | Causal, report FPS and per-frame latency (p50/p95) |
| "Degrade gracefully" | A **stress curve**: metrics vs. crowd density / occlusion level |
| Metrics: IDF1, ID-switches, MOTA, HOTA | IDF1/IDsw/AssA are the identity metrics; they care about identity more than detection |

**Takeaway:** this is an *algorithms + evaluation* challenge, not an app challenge. Top 5 out of 500+ will be decided by (a) numbers on standard benchmarks, (b) a novel, well-motivated occlusion-handling idea, (c) rigorous ablations and stress analysis, (d) a clean real-time demo.

## 3. Why Zed Digital cares (company context)
Source: [zed.digital](https://zed.digital) and their public LinkedIn/job pages.
- ZED Digital (Columbus, Ohio; offices in Montreal; team spread across India, Canada, Netherlands; founded 2005) builds the **ZIG Platform**, an "AI-powered hands-free ticketing and validation system for mobility networks and venues".
- Products directly tied to PS-2: **Fast Lane** (express entry at stadiums/venues), **hands-free validation** for bus/rail/BRT ("entry detected as the rider approaches the access point"), **Fare Evasion Control** ("AI-powered fare verification that detects unpaid journeys in real time while preserving passenger privacy"), **Passenger Counting**, crowd management.
- Their stack mentions (public profile): Docker, Kubernetes, microservices, Flask, Terraform, Jenkins, GitLab, Datadog, React.
- **So the "crowded gate" is literally their product scene.** A tracker that keeps IDs through occlusion lets them: link a person to a validated ticket all the way through the gate, detect **tailgating** (two bodies, one validation), and count people accurately.

**Pitch angle:** frame your tracker as the identity layer under ZIG: "every person crossing the gate keeps one ID from approach to exit, so validation, tailgating detection and counting are trustworthy." Also stress **privacy**: use body appearance embeddings kept in memory for a short window only, no face recognition, no stored identities. That mirrors their own privacy wording.

> Note: PS-1 (open-set re-ID with near-identical people) is the sibling problem. Good PS-2 work reuses those ideas for re-association, which is a nice story for judges.

## 4. Core concepts you must be fluent in
1. **Tracking-by-detection pipeline:** detector per frame → predict existing tracks (Kalman) → build cost matrix (motion IoU/Mahalanobis + appearance cosine) → assignment (Hungarian / `scipy.optimize.linear_sum_assignment` or `lap`) → update/birth/death.
2. **Kalman filter for boxes:** state usually `(x, y, a, h, vx, vy, va, vh)` (SORT/DeepSORT) or `(x, y, w, h, ...)` (BoT-SORT). Failure mode: during occlusion there is no measurement, error accumulates, and the track drifts.
3. **Appearance (re-ID) embeddings:** OSNet / FastReID-style backbone, cosine distance, track embedding updated by EMA. Failure mode: a half-occluded crop contains the *other* person, so the EMA gets polluted, which causes ID swaps.
4. **Low-score detections:** ByteTrack's key insight is that occluded people get low detector confidence, so match those too in a second round.
5. **Lifecycle:** `tentative → confirmed → lost (kept for max_age frames) → removed`. Re-association = matching new detections to *lost* tracks.
6. **Camera motion compensation (CMC):** gates are usually static cameras, but benchmarks (MOT17) have moving ones; BoT-SORT uses GMC.
7. **Metrics:**
   - MOTA = 1 − (FN + FP + IDSW)/GT (detection-dominated).
   - IDF1 = identity F1 over the whole sequence (identity-dominated).
   - HOTA = sqrt(DetA × AssA), balances both; **AssA** is the association part.
   - IDsw = number of identity switches.
   - Use the official **TrackEval** code, never your own metric implementation.

## 5. Literature map (all arXiv IDs verified)
### Foundations
- SORT, Bewley et al. 2016 · [arXiv 1602.00763](https://arxiv.org/abs/1602.00763) · Kalman + Hungarian on IoU.
- DeepSORT, Wojke et al. 2017 · [arXiv 1703.07402](https://arxiv.org/abs/1703.07402) · adds appearance + matching cascade.
- Performance Measures (IDF1), Ristani et al. 2016 · [arXiv 1609.01775](https://arxiv.org/abs/1609.01775)
- HOTA metric, Luiten et al. 2020 · [arXiv 2009.07736](https://arxiv.org/abs/2009.07736)
- CLEAR MOT metrics (MOTA), Bernardin & Stiefelhagen 2008, EURASIP JIVP (journal, no arXiv).

### Strong modern online trackers (your baselines)
- **ByteTrack**, Zhang et al. 2021 · [arXiv 2110.06864](https://arxiv.org/abs/2110.06864) · associate low-score boxes too. Must-have baseline.
- **OC-SORT**, Cao et al. 2022 · [arXiv 2203.14360](https://arxiv.org/abs/2203.14360) · fixes Kalman drift during occlusion (observation-centric re-update, momentum, recovery). **Most relevant to PS-2.**
- **BoT-SORT**, Aharon et al. 2022 · [arXiv 2206.14651](https://arxiv.org/abs/2206.14651) · better KF state, camera-motion compensation, IoU+ReID fusion.
- **StrongSORT**, Du et al. 2022 · [arXiv 2202.13514](https://arxiv.org/abs/2202.13514) · modernised DeepSORT, AFLink + GSI (offline parts: don't use those, PS is online).
- **Deep OC-SORT**, Maggiolino et al. 2023 · [arXiv 2302.11813](https://arxiv.org/abs/2302.11813) · *adaptive* re-ID weighting by detection confidence. Close to the "partial observation" theme.
- **Hybrid-SORT**, Yang et al. 2023 · [arXiv 2308.00783](https://arxiv.org/abs/2308.00783) · "weak cues" (confidence state, height) help in occlusion.
- **SparseTrack**, Liu et al. 2023 · [arXiv 2306.05238](https://arxiv.org/abs/2306.05238) · pseudo-depth scene decomposition for crowds. **Very relevant for dense gates.**
- **GHOST**, Seidenschwarz et al. 2022 · [arXiv 2206.04656](https://arxiv.org/abs/2206.04656) · simple re-ID handling for active vs. inactive tracks, long-term re-association.
- **SMILEtrack**, 2022 · [arXiv 2211.08824](https://arxiv.org/abs/2211.08824) · occlusion-aware similarity learning.
- **UCMCTrack**, 2023 · [arXiv 2312.08952](https://arxiv.org/abs/2312.08952) · ground-plane motion; good when camera is fixed (gates).
- **BoostTrack++**, 2024 · [arXiv 2408.13003](https://arxiv.org/abs/2408.13003) · tracklet info to recover missed detections.
- **DiffMOT**, 2024 · [arXiv 2403.02075](https://arxiv.org/abs/2403.02075) · non-linear motion prediction, real-time.

### Joint / end-to-end (know them, probably don't build them)
- FairMOT · [arXiv 2004.01888](https://arxiv.org/abs/2004.01888) · MOTR · [arXiv 2105.03247](https://arxiv.org/abs/2105.03247) · MOTRv2 · [arXiv 2211.09791](https://arxiv.org/abs/2211.09791). Heavier, slower to train, harder to explain; tracking-by-detection wins on time-to-results.

### Components
- Detector: YOLOX · [arXiv 2107.08430](https://arxiv.org/abs/2107.08430) (ByteTrack/OC-SORT use it; Apache-2.0).
- Re-ID: OSNet · [arXiv 1905.00953](https://arxiv.org/abs/1905.00953) (compact, real-time; matches the "compact backbone" spirit).

### Datasets
- **MOT17** (standard), **MOT20** · [arXiv 2003.09003](https://arxiv.org/abs/2003.09003) (very crowded, the closest public match to "crowded gate"), **DanceTrack** · [arXiv 2111.14690](https://arxiv.org/abs/2111.14690) (uniform appearance + heavy occlusion, so appearance alone fails; great stress test), **CrowdHuman** · [arXiv 1805.00123](https://arxiv.org/abs/1805.00123) (detector training for crowds), SportsMOT · [arXiv 2304.05170](https://arxiv.org/abs/2304.05170) (optional).

## 6. Our contribution and how it differs from prior work (revised after room review)
**Headline: bounded-latency, online multiple-hypothesis association for tight groups.**
When a track's best and second-best matches are within a margin, we don't commit. We keep both hypotheses open for at most *K* frames (fixed latency budget, e.g. K = 3 to 5), only for the ambiguous pairs, and resolve when the group separates or the window expires. The rest of the scene is matched greedily in the same frame as usual.

**Supporting contribution: inference-time visibility estimation.** Estimate how hidden each person is from what the tracker sees: box overlaps plus depth order (with a camera looking down at a gate, a lower box bottom usually means closer). Use it to scale appearance cost, freeze embedding updates, and widen motion gating. We **score the estimate against the ground-truth `gt.txt` visibility column** on MOT17/MOT20, which makes it a measurable contribution of its own.

| Prior work | What it already does | How we differ |
|---|---|---|
| Multiple Hypothesis Tracking: Reid 1979 (IEEE Trans. Automatic Control, "An Algorithm for Tracking Multiple Targets"); Kim et al., "Multiple Hypothesis Tracking Revisited", ICCV 2015 | Keeps a tree of association hypotheses and decides later, typically with deep windows and heavy pruning | Online and real-time: we only branch for ambiguous pairs (cost margin below a threshold), with a hard cap of K frames. The p95 and worst-case latency are reported as part of the method |
| Deep OC-SORT ([2302.11813](https://arxiv.org/abs/2302.11813)) | Scales appearance weight and embedding updates by detection confidence | We use an explicit visibility estimate from geometry and depth order (not confidence), validate it against GT visibility, and pair it with deferred assignment |
| Hybrid-SORT ([2308.00783](https://arxiv.org/abs/2308.00783)) | Uses detection confidence and box height as weak occlusion cues | Our cue is a per-pair occlusion relation (who is in front of whom), and it drives decisions, not just cost terms |
| SparseTrack ([2306.05238](https://arxiv.org/abs/2306.05238)) | Pseudo-depth layering for crowded matching | We use depth order to estimate visibility per person, not to split the matching into layers |
| OC-SORT ([2203.14360](https://arxiv.org/abs/2203.14360)) | Fixes Kalman drift after occlusion | We keep it as part of our motion backbone and ablate it separately |

**Novelty claim, worded honestly:** "a bounded-latency online variant of MHT, restricted to ambiguous associations and guided by an inference-time visibility estimate", not "a new idea".

**Comparisons we must show:** ByteTrack, OC-SORT, BoT-SORT, **Deep OC-SORT and Hybrid-SORT**, all on the same cached detections and embeddings. Report the ID-switch gain against the extra p95 latency for each K.

**Lower-priority extras (only if time allows):** gallery-based re-association that keeps a bank of high-visibility embeddings, and a direction-aware motion prior based on the gate's ground plane.

## 7. Evaluation plan that makes judges trust you
- **Baselines:** ByteTrack, OC-SORT, BoT-SORT (same detector weights for all, so the comparison is fair).
- **Benchmarks:** MOT17 val (half split), MOT20 val, DanceTrack val. Report HOTA, AssA, IDF1, IDsw, MOTA, FPS.
- **Stress analysis (explicitly required):** bin ID switches and IDF1 by GT visibility band (0–0.25, 0.25–0.5, 0.5–0.75, 0.75–1 from the `gt.txt` visibility column) and by people per frame. Plot IDF1/IDsw vs. density for every method. Also do **synthetic occlusion injection**: randomly drop detections for k frames (k = 5, 15, 30) and measure re-association success.
- **Ablations:** each component on/off, in a table.
- **Latency:** per-stage ms (detector / re-ID / association), p50, **p95 and max**, on the stated GPU, plus a CPU-only tracker number.
- **Online only:** no offline smoothing, interpolation or global linking in any reported number.
- **Test discipline:** tune on held-out halves of the training sets and DanceTrack val; MOT20 test submissions are limited, so use them only for final numbers.
- **Failure gallery:** 3–4 video clips showing where baselines swap IDs and yours doesn't (and one where yours still fails; honesty scores points).

## 8. Questions to confirm with the organisers
- Submission format: code repo, report, video, live demo? Page limits?
- Are specific datasets mandated, or is any public MOT benchmark ok? Is a private test set used?
- Hardware assumed for "real-time" (GPU class, FPS target)?
- Are pretrained detectors/re-ID weights allowed?
- Deadline and judging timeline.
