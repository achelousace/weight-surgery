---
name: weight-surgery
description: Find the weights that carry a behaviour in a trained transformer and edit them directly, without training ("weight-space programming", or "wsp"). Search attention heads (induction and copy scans, ablation and scaling scans, attention taps with masks and forced looks), feed-forward units and write matrices (stream directions, writer attribution). Separate two jobs inside one head (contrast directions), apply reversible edits (scale, cut or shrink directions, rank-1 removals with random controls, hooks), screen them on probes and damage, test generation with paired seeds, and save the chosen edit as a new, verified checkpoint with a manifest and an undo file. Use it whenever someone wants to know which heads, neurons or directions cause a behaviour (loops, repetition, a habit, a capability), switch off or scale heads, ablate or hand-edit weights, steer or project out a direction, compare an edited model with the original or save it, even if they only say "wsp", "weight edit", "head ablation" or "which head does this".
---

# Weight surgery

A behaviour you don't want (a loop, a habit) and the abilities you must keep are both computed by specific weights:
heads, feed-forward units, and directions inside them. This skill finds those weights in a trained transformer and
edits them directly, with no training. The steps are: search the sites, capture what they write, separate the
unwanted job from the wanted one, edit reversibly, screen every candidate, test generation, and save the winner as a
new checkpoint that is verified and can be undone.

The method comes from real studies on small chat models; `references/cases.md` walks through one with its numbers.
They found the copy heads behind repetition loops, separated their loop job from their name-copying job inside each
head, and saved a verified edit. They also found the limits. A probe that moves isn't a generation gain. And the heads
that *build* an ability can't be edited into a better one, because that is training's job.

## The pipeline

| # | Question | Method | Code (`scripts/surgery.py`) |
|---|---|---|---|
| 0 | What exactly is unwanted, and what must survive? | the model's own examples on fresh prompts, matched pairs, protected rows | `loop_onset`, `decision_positions` |
| 1 | Where do we stand, and how big is the noise? | onset and tail probes, damage rows, a capability metric, generation with paired seeds | `onset_probe`, `tail_probe`, `point_probs`, `rate_change` |
| 2 | Which sites carry it? | induction / copy scan; head ablation and scaling scans; attention taps; stream directions and writer attribution | `induction_scan`, `head_scan`, `AttnTap`, `mean_direction`, `writer_attribution` |
| 3 | Inside a site, which directions do which job? | capture at decision positions; contrast directions; energy shares; shared-job cosine | `capture_inputs`, `capture_outputs`, `contrast_directions`, `shares`, `orthonormal` |
| 4 | Which edit? | scale, project_in, project_out (rank-1 removal), runtime hooks, all reversible | `Surgeon` and its ops |
| 5 | Which candidate? | screen with dose series and a random control; the onset probe first; a damage limit | `screen`, `dose_series`, `exchange` |
| 6 | Does it change what the model writes? | fresh held-out prompts, greedy plus sampled rows, paired seeds, enough rows | your generator + `rate_change` |
| 7 | Keep, drop, or stop? | the decision rules below | |
| 8 | Save, and be able to undo | new file, verified diff, manifest, undo file | `save_edited`, `undo_saved` |

`references/methods.md` has each step in full: exact data construction, formulas, probe definitions, the edit
catalogue with model-layout notes, attention route diagnostics, saving, and every bug found so far. Read the section
for a step before running it on a real model.

## Stage notes

**0. Define the behaviour and build the data.** Write down three things:
- the unwanted behaviour, precisely. For example, a loop starts at the second occurrence of the first 4-token sequence
  seen three or more times (`loop_onset`).
- the wanted job that may share its machinery, such as copying names from the question into the answer.
- the abilities that must survive: general text, reading, the gold token's rank.

Then build three data sets:
- **The model's own examples:** generate on prompts never used in its training or any evaluation, greedy plus one
  sampled row and a short cap. Split them *by prompt* into a collect half (for directions) and a probe half (for tests),
  never both. Fail loudly if an exclusion file is missing; skipping it silently lets trained prompts back in.
- **Matched pairs:** one prefix with two continuations, the unwanted one (the last sentence repeated 4x) and the wanted
  one (the real rest), plus a row that closes early. Plain "onset minus start" averages mix *late in the answer* with
  *repeating*; matched pairs don't.
- **Protected rows:** fixed held-out rows per ability, drawn once by seed. Count how many are really scored (a nominal
  48 rows can turn out to be 44).

Rebuild the held-out sets for every new checkpoint, because what it trained on is no longer held out.

**1. Baseline and noise.**
- The onset probe gives P(the first repeated token) at real loop onsets and how often it is the top choice. A loop is
  flagged at its third repeat, so the onset token must stop being top-1 for greedy decoding to change.
- The tail probe gives mean log-prob per token of the repetitive and the productive continuation, plus the chance of
  an early close.
- Damage: loss, gold top-1 and mean rank on the protected rows.
- A capability metric, and a generation test.
- The noise of a rate is sqrt(p(1-p)/n): about 1.2 points for loops at 17% on 960 rows, and at most 50/sqrt(n). Use
  `rate_change`. Seed generation the same way for every model compared (paired sampling), so differences come from the
  weights, not the dice.

**2. Search the sites.**
- **Heads:**
  - `induction_scan`: on repetitive texts, the attention a repeated token pays to the position *after* its previous
    occurrence; that is the copy mechanism. Offset 0 gives the plain copy score. A true induction head scores high on
    the first and low on the second (for example 0.42 vs 0.012).
  - `head_scan`: every head off, one at a time, then the strongest at 0.5, 0.75, 1.25 and 1.5.
  - Rerun the scans on every checkpoint.
- **Stream and write matrices:**
  - the per-block mean difference of the block output between unwanted and wanted positions
  - `writer_attribution` for a write matrix: each unit's write along the direction times its activation contrast
    gives the top units
  - a layer sweep: a small edit per layer, read the tail probe, and pick the layer where the repetitive branch drops
    most while the productive one and the close barely move
- **Side channels or slots** (if the model has a memory or side channel): a z-score per slot,
  (mean unwanted - mean wanted) / std of wanted.
- **Attention routes:** `AttnTap` shows where the look lands. It can mask sink keys, force a one-hot look on the right
  key (this separates "can't find it" from "can't use it"), and centre near-constant queries.

**3. Inside a site, separate the jobs.**
- Capture what the site writes at the **decision positions**, the positions whose output predicts the token (prompt
  length - 1 + j, `decision_positions`). Capture separately for the unwanted job, the wanted job and ordinary text.
  Capturing one step late is an easy slip, and it blurs the contrast.
- `contrast_directions` gives generalised eigenvectors of C_unwanted v = lambda C_wanted v, using uncentred second
  moments ("what the head writes"), ordered by energy ratio.
- Report each direction's share of both energies, and the cosine between the jobs' mean outputs. A cosine near 1 means
  one head does both jobs and the separable part is thin; it was 0.65 to 0.82 for the copy heads in the worked case.
- The directions are not orthogonal, so always `orthonormal(V, k)` before cutting. An edit built on the raw directions
  is not a true projection.

**4. Edit, always reversibly** (the `Surgeon` keeps every touched tensor's original and reverts bit-exactly):

| op | what it does | formula |
|---|---|---|
| `scale` | a head off or scaled (its input columns of the output projection), a feed-forward unit (its column of the down projection), a write slot (its row) | `W[:, cols] *= f`, `W[rows] *= f` |
| `project_in` | cut (scale 0) or shrink k directions inside one head's output | `W[:, cols] <- W[:, cols] (I - (1-s) Q Q^T)`, Q orthonormal |
| `project_out` | remove (s 1), shrink (s < 1) or reverse (s > 1) what a matrix writes along a direction, for all units or chosen ones. k=1 with the top writers is a rank-1 edit | `W[:, units] <- W[:, units] - s Q (Q^T W[:, units])` |
| `project_out` + `control_seed` | the same edit along a random direction orthogonal to Q, rescaled to the same size: the control | |
| `hook` (`remove` / `add` / `scale_units`) | runtime activation edits for quick tests and steering; reversible but not saveable | `x - s (x.d) d`, `x + s d`, `x[..., u] * s + add` |

Remove directions from the **writing weights**, not from the stream. In the worked case a stream projection raised the
general loss by +0.32 to +0.75, while the same direction removed from the feed-forward write weights cost +0.001.
Adding a direction (steering) did nothing. A spec is a JSON list of ops; directions are referred to by name, so the
spec can go into the manifest as is.

**5. Screen** (`screen`): each candidate is applied, measured on fixed rows, reverted, checked bit-exact, and logged as
one JSON line.
- Run a **dose series** (alpha 1, 2, 4, 6): a real effect grows with the dose. In the worked case the repetitive tail
  went -0.12 / -0.30 / -0.54 / -0.79 / -1.24 per token.
- Run a **random control** of the same size, which must do about nothing (+0.0002 and -0.003 in the worked case).
- Filter on the onset probe first; "moved" can mean a top-1 share under 80%.
- Set the damage limit before you look, for example a general-loss rise of at most 0.24.

**6. Generation is the verdict.**
- Use fresh prompts never used anywhere, greedy plus sampled rows (T 0.5 and 0.7), and paired seeds, with enough rows:
  in the worked case every 40-prompt win vanished on 120 fresh prompts. Compare with `rate_change`.
- A probe shift is not a greedy flip. One edit cut sampled loops from 19.4% to 9.4% and left greedy unchanged.
- Read the greedy transcripts too.

**7. Decide.**
- Keep an edit only if generation moves past the noise at an acceptable cost.
- If the probes move but generation doesn't, the edit acts on the right mechanism but what is left of the problem is
  out of its reach. After further training (preference tuning) had already removed most loops, the same split still
  moved the onset probe, but its loop gain fell inside the noise.
- A head whose removal collapses an ability, that every scale makes worse, and that has no separating direction is a
  **foundation**, not a fault: there is no head edit to make.
- Hand edits of shared components can be real in isolation and destructive in the network. Hand those to training:
  the directions found here can still serve as a training region (aim a loss at them), though the training push may
  barely land there.

**8. Save and undo** (`save_edited`):
- It never writes the source and never overwrites a file.
- It builds the new file from the source (every other payload key kept, provenance in `extra.weight_surgery` or in
  safetensors metadata), keeping the source's tensors and dtypes.
- It reloads the new file and checks that exactly the edited tensors differ and the source's hash is unchanged.
- It writes `out.edit.json`: a manifest with, per tensor, the changed entries, rows and columns, the relative change
  and the before and after sha256.
- It writes `out.undo.pt`: the original tensors and the directions used.
- `undo_saved` rebuilds the original weights as a new file and checks them bit for bit.
- Runtime hooks can't be saved; bake them into weights with `project_out`.

## Wiring the harness (a Hugging Face model)

```python
import sys, torch
sys.path.insert(0, "path/to/weight-surgery/scripts")              # this skill's scripts folder
import surgery as su
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, attn_implementation="sdpa").eval()
cfg = model.config
L, H = cfg.num_hidden_layers, cfg.num_attention_heads
hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
o_proj = lambda l: f"model.layers.{l}.self_attn.o_proj.weight"      # head h = input columns h*hd:(h+1)*hd
fwd = lambda m, ids: m(torch.tensor([ids]))
logprobs = lambda m, ids: torch.log_softmax(m(torch.tensor([ids])).logits[0].float(), -1)

# damage, onsets, pairs, repetitive_seqs, batches, fwd_batch, select, X_loop, X_name, l and h are yours to build:
# references/methods.md sections 1 and 4 say how.
def tests(m):                                   # FIXED rows, same every call
    return {**damage(m), **su.onset_probe(m, logprobs, onsets),
            **{"tail " + k: v for k, v in su.tail_probe(m, logprobs, pairs).items()}}

scores = su.induction_scan(model, fwd, repetitive_seqs)            # {attention call (= layer): score per head}
base, off = su.head_scan(model, [(l, h) for l in range(L) for h in range(H)], tests, o_proj, hd)
X = su.capture_inputs(model, model.model.layers[l].self_attn.o_proj, batches, fwd_batch, select)   # [N, H*hd]
V, lam, s_bad, s_good, cos = su.contrast_directions(X_loop[:, h*hd:(h+1)*hd], X_name[:, h*hd:(h+1)*hd])
cands = {f"cut k{k}": [dict(op="project_in", param=o_proj(l), cols=[h*hd, (h+1)*hd], dirs="V", k=k)] for k in (4, 8, 12)}
base, rows = su.screen(model, cands, tests, dirs={"V": V}, log="screen.jsonl")
S = su.Surgeon(model, {"V": V}); S.apply(cands["cut k8"])
manifest = su.save_edited(model, "model.safetensors", "edits/model_cut_k8.safetensors", cands["cut k8"], S, state_key=None)
```

The layout notes are in `references/methods.md` §6: GPT-2's Conv1D needs `layout="in_out"`, grouped key/value heads
need care, and a norm after the output projection matters. Run `python scripts/surgery.py --selfcheck` once in a new
environment. It trains a tiny copy-task transformer on the CPU, which grows a real induction head, and asserts every
tool, including an exact revert, the projection removing the whole span, the random control matching size, the probes,
and a full save, refuse and undo cycle in both `.pt` and `.safetensors`.

## Rules learned the hard way

- **Orthonormalise before projecting.** On raw contrast directions, I - U U^T is not a projection for k >= 2.
- **Capture at the decision position**, not one step late.
- **Attribute a write matrix with its own input.** A mask built from another matrix's neurons picks the wrong units.
- **Check for a norm between the output projection and the residual.** There, zeroing a head rescales the others, and
  a projection is not an exact removal.
- **Keep data seeds separate from generation seeds.** A generation flag that re-seeds everything silently re-draws the
  probe set and the damage rows.
- **Keep probe data out of capture data,** or the probe is in-sample.
- **Use one definition of "loop"** in probes and generation (token 4-grams vs word 4-grams), or report both.
- **Map key indices through provenance every time.** A sink mask by index hits the wrong keys when the key order
  changes.
- **Centre queries, not keys.** Centring keys by a constant changes nothing after softmax.
- **Key the undo store by parameter name and verify every revert** (the `Surgeon` does both).
- **Remove from writing weights, not from the stream; removing beats adding.**
- **No CUDA graphs or compiled caches across weight edits.**
- **Delete big capture files** once you have the directions, and keep the scripts that rebuild them.
- **Read the screen data, not the summary.** A written summary can disagree with its own screen.
- **Never write the source checkpoint, never overwrite a file, verify the diff.**

## Reporting to the user

Plain English: which site, what edit, what changed, by how much, real or noise. Use one table per question. Name sites
in words ("head 0 of block 4", "layer 3 feed-forward"). Every number must come from a log or results file of this
run; say "I don't have that number" rather than estimate one. Give each edit's cost next to its gain. Don't recommend
saving an edit that generation hasn't confirmed.

## Files

- `scripts/surgery.py`: the harness and its probes (any PyTorch transformer; `--selfcheck`).
- `references/methods.md`: every method in full, model-layout notes, attention route diagnostics, saving, all bugs.
- `references/cases.md`: a worked case with its numbers, the bug review, and what was handed to training.
