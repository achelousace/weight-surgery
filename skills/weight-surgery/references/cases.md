# Worked case

One small decoder-only chat model, studied in two rounds (after supervised training, then after preference tuning),
plus one hand-off to training. All numbers come from the study's logs and reports. "Loops" = the share of generated
rows the loop checker flags. The protected abilities were general text ("general"), reading comprehension ("reading")
and multi-turn chat ("multi-turn"). The model also had a small side channel of memory slots, so slot edits were tried
too.

---

## Round 1: after supervised training

Goal: cut repetition loops without hurting reading and general text. It ran in four waves, 72 runs and 4.9 GPU-hours.

**Wave 1: a wide cheap screen** (300 natural loops; edits picked by hand; generation on 40 prompts).
- **Baseline:** loops 37.5%, onset probe 0.586 with top-1 89%.
- **Stream projection** (a hook removing the block direction): one layer cut loops to 22.5% but raised loss by +0.32.
  Another layer cost +0.75; four layers together cost +0.59 and broke reading.
- **The same direction removed from the writing weights:** one layer gave 32.5% at +0.001, all of it from the
  feed-forward down projection (the attention output projection alone did nothing). Three layers gave 27.5% at +0.11.
- **Steering** (adding the direction): nothing.
- **Side-channel slots:**
  - three slots zeroed: 27.5% at +0.13
  - one slot zeroed broke the model; doubling it gave NaN
  - one slot zeroed in one layer: 25% at +0.011
- **On 120 fresh prompts every win vanished:** baseline 40.0%, edits 41.7 / 39.2 / 40.0%.

Lessons: remove from the writing weights, not the stream; adding does nothing; small sets fool you.

**Wave 2: matched pairs, a writer direction, a rank-1 feed-forward edit.**
- **Tail probe baseline:** repetitive -1.15, productive -2.88, close -0.11, so this model closes readily.
- **Layer sweep:** one middle layer won (d_rep -0.123, d_prod -0.006, d_close +0.008).
- **Dose at that layer** (top 64 writers, alpha 1 / 2 / 3 / 4 / 6):
  - d_rep: -0.12 / -0.30 / -0.54 / -0.79 / -1.24
  - d_prod: -0.006 to -0.075
  - close: up to +0.024
- **Variants:** one half of the layer's units carried about 3/4 of the effect (-0.093 vs -0.029). Combinations added
  little and cost more.
- **Random orthogonal controls:** +0.0002 and -0.003, so the direction is real.
- **Generation:**
  - 40 prompts, alpha 1 / 2 / 4 / 6: loops 40 / 35 / 32.5 / 27.5%, reading loss +0.03 / +0.06 / +0.13 / +0.24
  - 120 prompts: greedy 43.3 and 42.5% against 40.0%, so no greedy gain
  - 160 prompts at alpha 6: all rows 29.4 to 27.1%, T 0.7 rows 19.4 to 9.4%, T 0.5 rows 25.6 to 29.4%, greedy unchanged

Lesson: a probe shift is not a greedy flip; the effect showed only under sampling.

**Wave 3: heads.**
- **Induction scan:** the top head scored 0.425 (copy score only 0.012, a true induction head); the next two scored
  0.252 and 0.123.
- **Feed-forward and slot edits** moved the onset probe only with large damage.
- **In the screen:**
  - the top head off: onset 0.529 / 84%, tail -1.08, general +0.11, reading +0.49
  - the top 3 off: 0.485 / 80%, tail -1.32, reading +0.61
  - the top 3 halved: tail -0.64, general +0.02, reading +0.14
- **Generation, 160 prompts:**
  - the top head off: loops 25.4% (greedy 40.6%), at general +0.31, reading +0.35 and top-1 -7.5 points
  - the top 3 off: 22.3% (greedy 38.1%, T 0.7 10.6%), at general +0.47, reading +0.43 and top-1 -10 points; the
    format errors rose from 28.7% to 35.2%
  - Both were rejected as too expensive.

Lesson: one head can do two jobs. The copy head also copies names into reading answers.

**Wave 4: the head split** (separating the loop job from the name job inside the copy heads).
- **Analysis, single head:** the first direction held 7.2% of loop energy against 0.9% of name energy (ratio 6.4).
- **k screen** (2 / 4 / 6 / 8 / 12 / 16 directions cut): tail -0.03 / -0.09 / -0.13 / -0.19 / -0.28 / -0.39,
  reading loss +0.04 / +0.06 / +0.07 / +0.08 / +0.12 / +0.17. The onset barely moved.
- **Generation, 160 prompts:** single head k 4: 25.2% against 29.4% (general +0.035, reading -0.004); three heads k 4:
  26.5%, k 8: 27.3%.
- **Generation, 320 fresh prompts (960 rows):**

  | | loops (all rows) | greedy | vs baseline |
  |---|---|---|---|
  | baseline | 30.5% | 41.6% | |
  | single head, k 4 | 27.9% | | 1.3 SE |
  | three heads, k 4 | 26.9% | | 1.8 SE |
  | three heads, k 8 | 25.2% | 35.3% | 2.6 SE; cost general +0.08, reading +0.01 |

**Verdict:**
- The method transfers, but the fix only partly does. Loops are flagged at the third repeat, so the first repeated
  token has to flip.
- The one component that flips it, the copy head, also copies names. The head split removed about a sixth of the loops.
- **Review note:** the split used raw contrast directions, which are not orthonormal, so for k >= 2 its edit was not a
  true projection (see "The review" below).

---

## Round 2: after preference tuning

Same harness, pointed at the tuned checkpoint, with fresh held-out sets. 133 of round 1's 320 evaluation prompts had
been used in tuning, so a new 320-prompt set was drawn, with 0 overlap. The model's own loops: 480 prompts, 960 rows,
272 looping, giving collect 136 and probe 100.

**Goal 1: the loop split, redone.**
- **The copy heads were unmoved by tuning:** 0.427 / 0.242 / 0.101.
- **Capture:** loop repetitions 2,479, productive 2,630, natural loops 2,176, names 457, reading other 4,653.
- **Directions:** the first held 7.1 / 4.5 / 8.3% of loop energy against 0.9 / 0.6 / 0.8% of name energy; 8 directions
  held 25 / 24 / 28% vs 8 / 6 / 5%. The cosine of the means was 0.82 / 0.65 / 0.82, so the jobs share the heads.
- **Baselines:** onset 0.574 / 95%; tail repetitive -1.658, productive -2.901; damage general 2.309, reading 0.945.
- **Screen:**

  | k | onset P / top-1 | tail rep / prod / close | loss change general / reading / multi-turn | reading score |
  |---|---|---|---|---|
  | 4 | 0.565 / 88% | -0.096 / -0.030 / +0.123 | +0.011 / +0.044 / +0.013 | 0.25 |
  | 8 | 0.545 / 87% | -0.190 / -0.037 / +0.106 | +0.017 / +0.062 / +0.025 | 0.25 |
  | 12 | 0.541 / 87% | -0.277 / -0.046 / +0.077 | +0.027 / +0.106 / +0.042 | 0.33 |

- **Generation, 960 rows:**

  | | baseline | k 4 | k 8 | k 12 |
  |---|---|---|---|---|
  | loops, all rows | 17.5 | 17.0 | 16.6 | 17.0 |
  | loops, greedy | 31.9 | 31.6 | 30.0 | 32.8 |
  | loops, T 0.5 | 15.0 | 15.0 | 16.6 | 12.5 |
  | loops, T 0.7 | 5.6 | 4.4 | 3.1 | 5.6 |
  | gold top-1, general | 55.2 | 55.0 | 54.7 | 54.4 |

- **Verdict:** the method transfers one for one and the probes move as before, but the generation gain doesn't.
  The changes are within the noise of 960 rows (about 1.2 points), because tuning had already removed the loops the
  split used to catch. The k 8 edit costs +0.017 general, +0.062 reading and -0.5 points of gold top-1.
- It was saved anyway as a new file (3 output-projection tensors) next to the source, and the source was verified
  untouched. Like round 1's, this saved edit used the non-orthonormal directions.

**Goal 2: the heads behind the gold token's ranking.**
- **Baselines:** general top-1 55.2%, mean rank 60.4, loss 2.309; reading 84.1% / 27.7 / 0.945; multi-turn 60.7% /
  25.0 / 1.764.
- **Ablation scan (every head off, 96 heads):**
  - No head raised general top-1 by as much as 0.2 points when removed; 46 moved it by less than 0.5 points either
    way. The two best "spreaders" (+0.2 points top-1, but +2.5 to +2.9 mean rank) were noise.
  - The biggest builders, general top-1 / rank / loss when removed:

    | head | top-1 | rank | loss |
    |---|---|---|---|
    | block 0 head 2 | -13.6 points | +98.7 | +0.989 |
    | block 0 head 0 | -11.4 | +72.5 | +0.735 |
    | block 0 head 1 | -10.3 | +61.6 | +0.737 |
    | block 1 head 0 | -9.3 | +31.0 | +0.552 |

  - Per block: block 0 -49.8, block 1 -28.4, block 2 -13.6, and later blocks near zero.
  - The copy head, switched off, cost general 2.1 and reading 7.6 points.
- **Scaling scan** (the 8 strongest at 0.5 / 0.75 / 1.25 / 1.5): every scale cost. Block 0 head 0: at 0.5, -3.9
  points, rank +15, loss +0.25; at 1.5, -3.2, +22, +0.20; at 0.75 and 1.25, -0.9 each.
- **Contrast split** (gold-on-top vs gold far down): the first-block heads had ratios of only 1.7-2.7, with means at
  cosine 0.977-0.999. Every top-1 change in the screen was within 0.3 points; the only effects past noise were costs.
  **Review note:** these positions were one step after the decision positions, so recheck before relying on "no
  separating direction".
- **Verdict:** the first blocks' heads build the ranking (half the top-1 rate is gone without block 0). They are a
  foundation, not a fault: no head edit improves it. The ranking drift during tuning is network-wide, which is
  training's business. No checkpoint was saved.

**The attention route probe (does the model look at a fact far back?).**
- **The setup:** a planted secret word far back in a long document, with the question at the end.
- **The look:**
  - It landed on the document opening: two structural tokens took 0.551 and 0.293 of the look.
  - The fact got 0.1-1.6%, and the first answer token sat at rank 3,680.
  - Forcing a one-hot look on the fact key gave rank 2, so the downstream route works.
- **The keys:** one input stripe made up 85-100% of each key's norm, and neighbouring keys had cosine 0.98-1.00, so
  keys were indistinguishable. The token embedding's stripe was under 1%.
- **The queries:** near-constant (cosine 0.90-1.00 between positions).
- **Fixes tried by hand:** masking sinks helped the plant (rank 166) but not natural facts. Centring queries made it
  worse in the network (rank 10,186), though it helped offline in single blocks. A question containing the fact's
  prefix "helped" only because the model copied it from the question.
- **Verdict:** no training-free rescue. Train the key's token stripe and the query with attention supervision, add a
  learned null key, and measure on this probe.

---

## Hand-off to training

A later training study asked whether an anti-repetition loss could be aimed only at the copy heads' loop directions
from round 2, leaving the copy mechanism intact. With the push restricted to the first 4, 8 or 12 loop directions, or
all 32 of the three heads' output columns, the measured effect over 10 steps was about zero on everything. For k 8:
general -0.0001, reading -0.0004, P(repeat) at loop onsets +0.0003. The training push barely lands there, so the loss
was placed by layer instead.

---

## The review (bugs found when the studies' code was reread)

- **The head-split edit, including both saved k 8 checkpoints, used non-orthonormal directions.**
  `I - U_k U_k^T` with C_name-orthogonal unit columns is not a projection for k >= 2; the energy shares over all 32
  directions summed to 0.55-0.63, not 1. The edits were real and screened, but they were not "8 directions cut".
  The fix is `project_in`, which orthonormalises first.
- **Capture positions were one step late** for names, natural onsets and the rank split. The first two blur the
  contrast. The third leaves goal 2's "no separating direction" verdict unproven.
- **A side-channel mask was picked from another matrix's neurons.**
- **The output projection fed a norm before the residual** in this model, so head edits were not exact removals there.
- **A generation seed flag silently re-drew the probe set and the damage rows.** The recorded baselines were not
  affected, but one head's onset read 0.529 in a screen and 0.493 in a later run.
- **The exclusion list skipped missing files silently,** and the files it named had since been renamed.
- **Damage scored 44 rows, not 48, and the reading score about 12.**
- **The tail probe was partly in-sample,** and the probe and generation used two different loop definitions.
- **A sink mask by index hit the wrong keys,** centring keys was a no-op, and the forced-look log showed the weights
  from before the force.
