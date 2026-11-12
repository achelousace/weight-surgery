# Methods in depth

Every method of the weight-surgery pipeline, as used in real studies (see `cases.md`), written for any transformer and
with the corrections a code review found. Function names refer to `scripts/surgery.py`. Numbers marked "example" come
from the worked case; use them for scale, not as targets.

Contents
1. Defining the behaviour and the data
2. Probes and tests
3. Searching for sites
4. Capture mechanics
5. Directions
6. The edit catalogue, and mapping it onto any architecture
7. Screening protocol
8. The generation verdict
9. Attention route diagnostics
10. Saving, manifest, undo
11. Pitfalls and bugs, with fixes

---

## 1. Defining the behaviour and the data

**1.1 The model's own examples.** Read the behaviour off the model itself, not off someone else's samples:
- **Prompts:** take them from the training pool, minus every excluded id (§1.5), and drop long ones (for example over
  1024 tokens). Draw a fixed mix by ability, for example 1/2 general chat, 1/4 reading, 1/4 multi-turn.
- **Generation:** greedy plus one row at T 0.7, top-p 0.95, a short cap (160 new tokens; a loop shows within about
  100), stopping at the end token or the turn's stop string.
- **Flag:** `loop_onset(ids, n=4, repeats=3)` gives the index of the second occurrence of the first 4-token sequence
  seen at least 3 times. Keep samples whose onset is at least 2.
- **Split by prompt id:** shuffle the distinct prompt ids, then collect = loops from the first half (capped at 300) and
  probe = loops from the other half (capped at 100). No prompt is in both.
- **Never redraw the sample file.** That would change the probe set. Guard it with an "exists, delete it first" check.

Example: 480 prompts, 960 rows, 272 looping (28.3%), giving collect 136 and probe 100.

**1.2 Matched pairs** (unwanted vs wanted continuation of one prefix). Built from gold answers:
- **Answer filter:** the prompt is at most 900 tokens, the answer body is at least 300 characters with no markup.
- **Sentences:** split on `(?<=[.!?])\s+` and keep those over 20 characters; a pair needs 6-16 sentences.
- **Rows,** with k = n // 2:
  - head = the answer opener of your chat format + the first k sentences + `" "`
  - productive = the remaining sentences
  - repetitive = (sentence k-1 + `" "`) x 4
  - close = head + your format's end-of-answer and end-of-turn markup

16 pairs: 8 "find" pairs for directions and 8 "transfer" pairs for the probe. Keep the probe pairs out of every capture,
or the tail probe is in-sample. Tokenise the prefix and continuation the way the model will see them joined, not
separately (a trailing space on the head tokenises differently).

**1.3 Position sets.** Always the **decision position**: the position whose output predicts the token, which is
prompt length - 1 + j for answer token j (`decision_positions`). Capturing at prompt length + j is one step late.
- **tail:** the last 36 tokens of a branch
- **onset:** the first 8 tokens of a branch
- **loop repetitions:** with s = branch start, e = branch end and per = max(1, (e - s) // 4), the positions s + per to
  e, i.e. the 2nd to 4th repeats
- **natural loop:** from the onset to onset + 15, in the model's own loops
- **clean:** the first answer tokens, matched in count to the onset set
- **wanted job ("name"):** reading answers' tokens that also occur in the last 256 prompt tokens and decode to
  `^ ?[A-Z][a-z]{2,}`, in rows with at least 2 such tokens
- **reading other:** every other answer position of those rows
- **productive:** the whole productive branch

Example sizes: loop repetitions 2,479, productive 2,630, natural loops 2,176, names 457, reading other 4,653.

**1.4 Protected rows (damage).** Use fixed held-out rows per ability group (for example general, reading,
multi-turn), at most 1024 tokens, drawn once by seed, with one forward per row. Per group:
- loss = sum of answer-token CE / number of answer tokens
- top-1 = share of answer tokens where the gold is the top choice
- mean rank = the number of logits strictly above the gold logit, 0-based
- log P(end) on rows ending in the end token

Count the rows actually scored: integer division and empty rows can take a nominal 48 down to 44.

**1.5 Held-out discipline.** The exclusion set is the union of:
- every prompt sampled or trained in any round
- the trainer's own held-out prompts
- every previous evaluation set
- the loop-sample prompts

Fail loudly when an exclusion file is missing. A harness that skips missing files silently (or names files that were
later renamed) loses its training exclusion without a warning. Reuse an existing evaluation file rather than
redrawing it. Rebuild for every new checkpoint: in the worked case, 133 of 320 evaluation prompts were later used in
preference training, so the set was "burnt".

## 2. Probes and tests

**2.1 Onset probe** (`onset_probe`). Teacher-force each held-out loop up to and including its onset token. Read
p = P(the onset token | everything before) and its rank. Report the mean p, the median, the top-1 share and the mean
rank. Greedy decoding loops whenever the onset token is top-1, so the top-1 share is the number to move.

Example baselines: mean p 0.586 with top-1 89%; after preference tuning 0.574 with 95% (the loops that survive
tuning are the committed ones).

**2.2 Tail probe** (`tail_probe`). Uses the transfer pairs, 3 rows each, in one batched pass:
- productive and repetitive = mean over pairs of the mean log-prob per token of the branch
- close = log P(the close's first token) right after the head

Screens report d_rep (more negative = repetition suppressed), d_prod (should stay near 0) and d_close (positive = early
closing more likely).

Example baselines: rep -1.147, prod -2.876, close -0.112 (a model that closes readily).

**2.3 Damage:** §1.4.

**2.4 A capability metric** guards the ability the wanted job serves. The worked case used a reading score: the first
content token of each answer correct when the fact sits far back in the context. It had about 12 rows, so it moved 8.3
points per row and only served as a coarse alarm. Prefer a metric with enough rows to resolve the change you care about.

**2.5 Generation test** (§8).

**2.6 Statistics.**
- SE of a rate = sqrt(p(1-p)/n), at most 50/sqrt(n) points.
- Compare two models with `rate_change` (two-proportion z). The worked case treated 2.6 SE as past the usual doubt line.
- Paired seeds and several rows per prompt make the samples correlated, so an unpaired SE is only a guide. The noise
  floor between two nearly identical models is the honest check.
- Small sets fool you: every win on 40 prompts vanished on 120.

**2.7 Budget.** Count every teacher-forced forward and every generation run as a pass, and log passes and seconds per
candidate. Example: a screen candidate cost 58 passes (13 onset + 1 tail + 44 damage), and a full generation run 59.

## 3. Searching for sites

**3.1 Induction / copy scan** (`induction_scan`, `induction_score`). Run repetitive texts through `AttnTap`, which
records every attention call's probabilities. For each repeated position t with previous occurrence prev:
- induction = a[h, t, prev + 1]: it looks at what came next last time, in order to copy it
- copy = a[h, t, prev]: it looks at the previous occurrence itself

Average over positions and rank heads by induction. If your model attends in chunks or over a cache, decide whether
the previous occurrence may sit in an earlier chunk, and include those keys when they matter.

Example: the top head scored 0.425 (copy 0.012, a true induction head), the next ones 0.252 and 0.123, then 0.091 and
the rest under 0.05. The scores were the same after further fine-tuning (0.427 / 0.242 / 0.101).

**3.2 Head ablation scan** (`head_scan`). Each head's input columns of the output projection are multiplied by 0, one
head at a time, and damage is measured per group. A head whose removal *improves* the gold's top-1 or rank would be a
"spreader", an edit candidate. A head whose removal destroys them is a builder. Example cost: 96 heads x 44 passes.

**3.3 Head scaling scan:** the strongest heads at 0.5, 0.75, 1.25 and 1.5. If every scale in both directions costs,
the head sits at its optimum.

**3.4 Stream direction per block.** d = mean(block output at onset) - mean(at clean positions), with diagnostics: the
mean cosine of the outputs with d at onset vs clean positions (example: +0.50 vs +0.17). This is better done with
matched pairs (3.6).

**3.5 Side-channel slots** (models with a memory or side channel). Per slot, z = (mean at onset - mean at clean) /
(std at clean + 1e-6). Keep the top slots by |z|.

**3.6 Writer attribution** (`mean_direction`, `writer_attribution`). Per block, from matched pairs:
1. u = unit(mean over rows of (block output, repetitive) - (block output, productive)), averaged over the tail
   positions. Onset and natural variants use the onset positions and the natural loops.
2. writer = u^T W_down: each unit's write along u.
3. dact = the mean activation contrast of each unit (repetitive - productive) at **W_down's own input**.
4. attr = dact x writer, and the writers are the top 64 by signed attr.

Diagnostics: the attribution share of the top 64 and `writer[top].norm() / writer.norm()`. Always attribute with the
edited matrix's own input; a mask built from another matrix's neurons picks the wrong units.

**3.7 Layer sweep:** for each layer, a small edit (alpha 1, the top 64, or all units) and the tail probe. Pick the
layer where repetitive drops most while productive and close barely move. Example: the winning layer gave d_rep -0.123,
d_prod -0.006, d_close +0.008.

**3.8 Candidate screens.** Cross sites with doses and direction kinds:
- feed-forward and side-channel writes, by direction (onset, tail, natural), layer and alpha 2, 4 or 6
- each of the top heads off, the top 3 off, the top 3 at 0.5
- multi-layer sets

The worked case ran 66 candidates, then 45 more. §7 has the screening protocol.

**3.9 Order of work that served:**
1. A wide cheap screen.
2. Matched pairs and a direction.
3. A layer sweep.
4. A dose series, sub-region and combination variants, and a random control.
5. Generation on a fresh set.
6. Then deeper: the head scan, the capture, the directions, the k screen, and generation at 160 and 320 prompts.

## 4. Capture mechanics

- Use a **forward hook** on a block for its output, and a **forward pre-hook** on a matrix for its input. The
  attention output projection's input holds every head's output side by side; the down projection's input is the
  feed-forward activation. `capture_inputs` and `capture_outputs` do both and select (batch, position) pairs.
- **Chunked or streaming forwards** call a module once per chunk. The harness joins the calls along the time axis. If
  you map positions to (chunk, offset) by counting calls, a tap that counts calls needs one pass per position set.
- Store float32 on the CPU, run under no_grad (fp16 autocast is fine), in batches of 8, skipping rows over the memory
  limit.
- Keep the sizes in the log: the counts per set (§1.3) are part of the result.
- Delete big capture files once the directions are taken, and keep the script that rebuilds them.

## 5. Directions

**5.1 Mean difference** (`mean_direction`): unit(mean unwanted - mean wanted). With matched rows this is the mean of
paired differences.

**5.2 Writer attribution** (§3.6).

**5.3 Contrast directions** (`contrast_directions`). For one head's slice (32 dimensions in the example), in float64:
- C_bad = X_bad^T X_bad / N and C_good = X_good^T X_good / N. These are uncentred second moments, "what the head
  writes".
- C_good += eps I, with eps = 1e-3 x mean(diag C_good).
- W = C_good^(-1/2) from eigh; M = W C_bad W; eigh gives (lambda, E), sorted descending; U = W E with unit columns.
- lambda_i is the energy ratio u^T C_bad u / u^T C_good u.

Report each direction's share of the unwanted, wanted, productive and other energy (`shares`), and
cos(mean bad, mean good). **U is C_good-orthogonal, not orthonormal.** Its cumulative shares don't sum to 1 (0.55-0.63
over all 32 in the example), and I - U U^T is not a projection for k >= 2. Always use `orthonormal(U, k)` (QR) before
an edit.

Example: in three copy heads the first direction held 7.1 / 4.5 / 8.3% of loop energy against 0.9 / 0.6 / 0.8% of name
energy (ratios 6.8, 12.0, 11.7). 8 directions held 25 / 24 / 28% vs 8 / 6 / 5%. The cosine of the means was 0.82 /
0.65 / 0.82: the two jobs share the head, and the loop-only part is thin.

**5.4 Random control direction:** r is random (seeded), with r <- r - Q Q^T r and then unit length. The control edit
is the real edit with r in place of u, on the same units, rescaled to the same Frobenius norm
(`project_out(..., control_seed=s)`).

**5.5 A contrast aimed at a capability.** For example, the directions separating positions where the gold is top-1
from positions where it is at rank 10 or worse: capture both at the **decision** positions. Example (taken one step
late, so recheck before relying on it): the first-block heads had ratios of only 1.7-2.7 with cos(means) 0.977-0.999,
so there was no separating direction.

## 6. The edit catalogue, and mapping it onto any architecture

**6.1 The ops** (a spec is a JSON list; directions are named entries in `dirs`):

| op | formula | typical use |
|---|---|---|
| `scale` cols / cols_idx | `W[:, cols] *= f` | a head off (f 0), halved (0.5), boosted; a feed-forward unit |
| `scale` rows / rows_idx | `W[rows] *= f` | a write slot (a side-channel row) |
| `project_in` | `W[:, cols] <- W[:, cols] (I - (1-s) Q Q^T)`, Q orthonormal | cut k directions inside a head |
| `project_out` | `W[:, units] <- W[:, units] - s Q (Q^T W[:, units])` | remove a direction from a write matrix (s 1, all units), or the rank-1 edit (k 1, top-64 units, s = alpha) |
| `project_out` + `control_seed` | the same along a random orthogonal direction, norm-matched | random control |
| `hook` remove | `x <- x - s (x.d) d` at a module's output | a stream projection |
| `hook` add | `x <- x + s d` | steering |
| `hook` scale_units | `x[..., u] <- x[..., u] * s + add` | a slot hook |

What the dose means for `project_out`: s = 1 removes those units' write along u. s = 6 leaves (1 - 6) = -5 times it,
so the units then write *against* u.

A sensible default is never to edit norms, embeddings, the output head or whole tensors.

**6.2 Where things live in common architectures:**
- **Llama, Mistral, Qwen, Gemma** (Hugging Face):
  - `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight` and `model.layers.{i}.mlp.{gate,up,down}_proj.weight`
  - `o_proj.weight` is [hidden, heads x head_dim]; head h is input columns h x hd to (h+1) x hd
  - `q_proj` rows h x hd onward belong to query head h
  - `down_proj.weight` is [hidden, d_ff]; feed-forward unit j is column j
  - `up` and `gate` rows are the unit's input side
- **Grouped key/value heads:** k and v have n_kv heads, and kv head j serves query heads j x r to (j+1) x r - 1
  (r = heads / n_kv). Editing one kv head touches all of its query heads. `AttnTap` repeats k and v to match.
- **GPT-2 (Conv1D):** weights are stored [in, out]. Pass `layout="in_out"` and give `cols` in input dimensions:
  - `attn.c_proj.weight`: head h is rows h x hd onward of the stored weight
  - `mlp.c_proj.weight` [d_ff, d]: unit j is row j of the stored weight
  - `attn.c_attn` fuses q, k and v along its output
- **A norm after the output projection** (for example Gemma's post-attention and post-feed-forward norms, or a norm on
  a side channel): zeroing a head rescales the others, and a projection is no longer an exact removal. Measure, don't
  assume.
- **Tied embeddings:** editing the input embedding edits the output head too.
- **Biases:** column edits leave the projection's bias alone, so a "head off" still leaves the bias.
- **Precision:** edit in fp32 and cast back to the parameter's dtype. bf16 rounding can swallow tiny edits, and
  `save_edited` counts changed entries after rounding and refuses an edit that rounds away completely.
- **Attention capture:** load with `attn_implementation="sdpa"` so `AttnTap` sees the calls. With eager attention, use
  `output_attentions=True`. Flash-attention kernels bypass SDPA.
- **No CUDA graphs or compiled caches** across weight edits; run the sampler eagerly.

## 7. Screening protocol

1. Measure the baseline once per process: onset probe, tail probe, damage, and the capability metric.
2. For each candidate: apply, measure, revert in `finally`, and check the revert is bit-exact (`screen` does all of
   this).
3. Log one JSON line per candidate: tag, spec, seconds, passes, the absolute numbers and the deltas.
4. Read the screens in this order:
   - the onset probe first (for example "moved" = top-1 share under 80%)
   - the tail: repetitive down a lot, productive near 0, close not up
   - damage within the limit you set beforehand (for example general loss +0.24)
5. Before trusting a direction, run a dose series and a random control. Then try sub-region variants (in the example,
   one half of a layer's units carried about 3/4 of the effect) and combinations, which usually add little and cost
   more.
6. Send the 1-3 best to generation. Read the JSON data, not a summary: a summary can disagree with its own screen.

## 8. The generation verdict

- **Prompts:** fresh (§1.5), at most 2048 tokens, a fixed mix by ability (for example 1/2 general, 1/4 reading, 1/4
  multi-turn), drawn round-robin one per dataset. 320 prompts gave 960 rows.
- **Cap:** max(24, min(160, 1.5 x gold length + 32)). The no-ending rate depends on the cap, so compare it only at the
  same cap.
- **Rows:** greedy plus T 0.5 and T 0.7 with top-p 0.95. Paired seeds: `torch.manual_seed(SEED)` right before building
  the sampler for each model. Run eagerly.
- **Labels:**
  - loop: a word 4-gram 3 or more times in at least 12 words, or 3 or more consecutive tag-only lines. This is a
    different definition from the probe's token 4-grams, so report both or align them.
  - whatever other failures your evaluation counts: no ending (only when the gold ends with the end token), broken
    format, refusals, wrong facts, filler
  - repeat ratio = 1 - distinct 4-grams / (n - 3)
- **Report per mode** (greedy, each temperature): an edit can cut sampled loops and leave greedy unchanged. Save the
  greedy transcripts and read some of them.
- **Compare** against a seed-matched baseline run in the same setup, with `rate_change`, and against the noise floor.

## 9. Attention route diagnostics

For "does the model look at the right thing, and can it use it?":
- **Record and intervene** with `AttnTap`: probabilities per call; `mask_keys` (-inf); `force` (the output at a
  position becomes v[key], a forced one-hot look); `centre_q` (subtract the causal running mean of the queries).
- **Provenance:** if keys come from a cache, a memory or a retrieval step, wrap it to map every key to its source
  (document, position, token). Index masks are valid only while the key order is unchanged, so recompute them per
  document. A sink mask by index hits the wrong keys when the order changes.
- **Needle documents:** plant a fact at a known distance, then build distance variants (near, middle, far), cue
  variants (the question repeating the fact's prefix is a confound: the model copies from the question), and natural
  facts with no plant. Answer ids must re-tokenise consistently with the prompt.
- **Mass accounting at the decision position:** mass on far vs near context, on the fact's copies, the planted
  sentence, cue words, and structural tokens (sinks); entropy; the share of heads whose top key is the fact; the fact
  key's rank under the real query.
- **Stripe attribution of a fused projection:** when a key is built from several inputs concatenated, put a pre-hook
  on its input, with k = sum over stripes of x_stripe W_stripe^T + residual (an adapter delta). Report per stripe the
  norm share, the score share (sum |part . q| / sum |k . q|, which can exceed 1 through cancellation), and the cosine
  with neighbouring keys' same part. The stripe whose neighbour cosine is near 1 makes keys indistinguishable.
- **Query diagnostics:** the cosine between consecutive queries, and the shared mean's share of query energy. A
  near-constant query lets generic sinks win.
- **Offline sweeps from one stored pass:** centring, removing the top principal component, stripe gains and masks,
  scored without new forward passes. Centring keys is a no-op under softmax; centre queries.
- **Example findings:**
  - The look landed on the document opening: two sink tokens took 0.551 and 0.293.
  - Forcing a one-hot look on the fact key gave the answer rank 2, so the route downstream works.
  - One input stripe was 85-100% of each key's norm, with neighbour cosine 0.98-1.00, so keys were indistinguishable.
  - The query was near-constant (cosine 0.90-1.00).
  - Masking sinks didn't generalise to natural facts.
  - The verdict was "no training-free rescue": train the key's token stripe and the query with attention supervision,
    and add a learned null key. That is training's job.

## 10. Saving, manifest, undo

**`save_edited(model, src, out, spec, surgeon)`:**
1. Refuse if out = src, if out exists, if the formats differ, or if runtime hooks are active.
2. Hash the source.
3. Build the new file from the source:
   - a .pt payload: every other key kept, `extra.weight_surgery = {spec, source, source_sha256, note}`
   - a .safetensors file: the same in its metadata
4. Keep the source's tensor set and dtypes.
5. Reload the new file and check:
   - the differing tensors are exactly the edited ones
   - nothing changed outside the surgeon's touched tensors
   - the source hash is unchanged
6. Write `out.edit.json`: per tensor shape, dtype, changed entries, changed rows and columns, relative change, and base
   and edited sha256, plus totals.
7. Write `out.undo.pt` with the original tensors and the directions used.

**`undo_saved(edited, out, source=...)`** rebuilds the originals as a new file and checks them bit for bit against the
source.

For a sharded Hugging Face directory, `save_pretrained` to a new directory and diff the two state dicts the same way.

Example: cutting 8 directions in 3 heads changed 3 output-projection tensors, saved next to the untouched source.

## 11. Pitfalls and bugs, with fixes

| # | What went wrong | Fix |
|---|---|---|
| 1 | A head split projected with raw contrast directions; I - U U^T is not a projection for k >= 2 (a saved edit too) | orthonormalise (QR) first; `project_in` does |
| 2 | Captures at the token's input position, one step after the decision | `decision_positions`: prompt length - 1 + j |
| 3 | A side-channel writer mask built from another matrix's neurons | attribute with the edited matrix's own input |
| 4 | A norm after the output projection: head zeroing rescales the others; projections are not exact removals | check the block's code; measure, don't assume |
| 5 | A generation flag re-seeded everything: the prompt draw, the loop split (probe set) and the damage rows | separate data seeds from generation seeds |
| 6 | Exclusion files missing were skipped silently | fail loudly |
| 7 | The tail probe pairs were also in the capture | keep probe pairs out of capture |
| 8 | Two loop definitions (token 4-grams in the probe, word 4-grams in generation) | align them or report both |
| 9 | A sink mask by key index hit the wrong keys when the key order changed | map indices through provenance every time |
| 10 | Centring keys did nothing (constant shift under softmax) | centre queries |
| 11 | The undo store was keyed by class-specific names and nothing was checked after a revert | key by parameter name; verify bit-exact (the `Surgeon`) |
| 12 | A "top units" switch was a flag, not a count; a weight projection had no strength | `cols_idx` any size; `strength` on `project_out` |
| 13 | "48" damage rows were 44; the capability metric had about 12 rows | count rows scored; size metrics to the change sought |
| 14 | The attention scan kept only keys inside one chunk and relied on explicit masks | `AttnTap` adds the causal mask; include earlier context when it matters |
| 15 | A report's text contradicted its own screen data twice | read the data |
| 16 | A documented metric (an entropy, an "exact match") was never computed | check every reported field is really computed |
| 17 | Stream projection broke the model (+0.32 to +0.75 loss); steering did nothing | remove from writing weights instead |
| 18 | Probe wins that didn't survive generation, small-set wins that vanished on fresh prompts | generation on enough fresh rows decides |
