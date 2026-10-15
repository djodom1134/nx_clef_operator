# M0 results: Clef timing and accuracy on the dev site

2026-10-14 · Clef server on the DGX (`192.168.87.134:8100`), clips from Ubuntu_main (Nx Meta 6.1.2). Raw rows: `results/m0_timing.jsonl`, `results/m0_resolution.jsonl`, `results/m0_accuracy.jsonl`.

## Summary

- **Model time scales with frames, not questions.** About 75 to 125 tokens per frame and about 0.53 s per 1,000 tokens. A 30 s window costs 2.2 s at 1 fps, 3.7 s at 2 fps, 7 s at 4 fps, 11 s at 6 fps and about 26 s at 10 fps.
- **One camera at 10 fps uses more than the whole GPU.** A 30 s window at 10 fps takes 26.6 s of model time, longer than the 20 s between windows. The PRD's 10 fps ceiling only works for short bursts.
- **Shrinking frames is free accuracy-wise and saves about a third.** At about 113k pixels (448 × 252) a frame costs about 75 tokens; smaller frames cost the same. Shrinking a 640 × 360 clip to that size cut tokens by 33% and time by 34%, with identical person accuracy and more confident nuisance answers. The gateway now shrinks to this size by default.
- **Person detection is reliable on this footage.** On 29 hand-labelled motion clips (7 with people), Clef was right on all 29 at every setting tested, with mean P(person) 0.98 on clips with people and 0.01 without. 1 fps was as good as 4 fps.
- **Nuisance identification works for common causes.** Trees and plants in wind, light changes and headlights were usually named correctly; a waving flag was called "trees or plants" every time. Top-1 accuracy on 19 nuisance clips: 58% at 2 fps, 74% at 1 fps. Adding the site's own nuisance list to the state (as the PRD specifies) should fix cases like the flag.
- **Not yet tested: fast action.** The archive has no fights, falls or weapons, so whether 8 to 10 fps helps fast events is still open. That needs staged clips or a public fight dataset.

## Timing: 30 s live windows, fps sweep

Median of 2 runs per camera; secondary streams are 640 × 360 (H.264/H.265, about 6.4 to 7 fps), primary up to 2560 × 1440 at 25 fps.

| Stream | fps | Tokens | Model time (s) | Windows per 20 s step that fit in 90% of the GPU |
| --- | --- | --- | --- | --- |
| Secondary | 1 | 3,750 | 2.2 | 8 |
| Secondary | 2 | 7,050 | 3.7 | 4.9 |
| Secondary | 4 | 13,650 | 7.0 | 2.6 |
| Secondary | 6 | 20,250 | 11.0 | 1.6 |
| Primary | 8 | 30,690 | 21.4 | 0.8 |
| Primary | 10 | 38,250 | 26.6 | 0.7 |

Primary streams cost about 4 s extra per request (decoding 1080p to 1440p H.265 on the Clef server), 7 s on the 2144 × 2144 Garage camera. The gateway now decodes and shrinks clips itself, while the model works on the previous job, so that cost moves off the GPU's critical path.

Secondary streams on this site run at 6.4 to 7 fps, so questions above 6 fps need the primary stream.

## Resolution: same clips, re-encoded

| Frame size | Tokens at 2 fps | Model time at 2 fps (s) | Tokens at 6 fps | Model time at 6 fps (s) |
| --- | --- | --- | --- | --- |
| 640 × 360 | 7,260 | 3.8 | 20,460 | 10.8 |
| 448 × 252 | 4,904 | 2.5 | 13,544 | 6.8 |
| 320 × 180 | 4,904 | 2.5 | 13,544 | 6.8 |
| 640 × 640 | 8,117 | 4.2 | 23,477 | 12.6 |
| 448 × 448 | 6,317 | 3.3 | 18,077 | 9.3 |
| 320 × 320 | 4,757 | 2.6 | 13,397 | 6.7 |

The floor is about 0.1 megapixels per frame. The gateway shrinks every clip to at most 448 × 252 worth of pixels (`MAX_PIXELS`), keeping the aspect ratio.

## Accuracy: 32 archive motion clips, hand-labelled

Clips are 10 to 30 s of secondary stream from 8 cameras, about a quarter at night (IR). Labels are in `bench/labels.json`; 3 clips were too unclear to label for people.

| Setting | Person accuracy (n = 29) | Mean P(person): with / without | Nuisance top-1 (n = 19) | Median tokens | Median time (s) |
| --- | --- | --- | --- | --- | --- |
| 2 fps, native size | 1.00 | 0.98 / 0.01 | 0.58 | 4,457 | 2.66 |
| 2 fps, 448 px | 1.00 | 0.98 / 0.01 | 0.58 | 2,976 | 1.75 |
| 1 fps, 448 px | 1.00 | 0.98 / 0.01 | 0.74 | 1,768 | 1.21 |
| 4 fps, 448 px | 1.00 | 0.98 / 0.01 | 0.58 | 5,460 | 2.83 |

Nuisance answers by cause:

| Labelled cause | Clips | What Clef said most often |
| --- | --- | --- |
| Trees or plants in wind | 11 | Trees or plants in wind; night IR vines sometimes "insects or a spider web near the lens" |
| Lights changing | 3 | Lights changing (0.52 to 0.77); one basement clip "nothing visible" |
| Shadows moving | 2 | "Nothing visible" |
| Headlights | 1 | Headlights or "a vehicle" |
| Insects near the lens (night) | 1 | "Rain, snow or fog": streaks look alike |
| Flag waving | 1 | "Trees or plants in wind", every setting |

## What changes in the design

1. **Default frame size: about 0.1 MP.** Done in the gateway.
2. **Use 10 fps only in bursts.** A sustained 10 fps question on one camera cannot keep up. Proposal: run fast-action questions at 2 to 4 fps on every window, and switch a camera to 8 to 10 fps on the next window only when motion onset happens or the low-rate answer crosses half the threshold. This needs the fast-action test above before it is built.
3. **Capacity planning numbers.** At 2 fps with 0.1 MP frames, one window costs about 2.5 s, so one DGX handles about 7 camera-groups every 20 s at full load. Motion gating raises that a lot on quiet sites, since still windows skip questions that need motion.
4. **Nuisance learning should lean on the site's list.** Clef confuses similar movers (flag vs trees, insects vs rain). The site's known-nuisance list goes into every state, and mask proposals should require the same answer several times, as the PRD already says.
