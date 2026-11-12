#!/usr/bin/env python3
"""surgery.py -- the weight-surgery harness (read ../SKILL.md first).

Find the weights that carry a behaviour in a trained transformer and edit them directly, without training:
  search    induction / copy scans, head ablation and scaling scans, attention capture with interventions
  capture   what a module reads or writes at chosen positions (the decision position: the one whose output predicts the token)
  directions  mean difference, writer attribution (top units), contrast directions (generalised eigen), random controls
  edits     reversible ops on any weight: scale (heads, units, slots), project_in (cut directions inside a head),
            project_out (remove / shrink / reverse what a matrix writes along a direction; the rank-1 edit), runtime hooks
  probes    loop_onset, point_probs, onset_probe, tail_probe, rate_change, exchange, table
  screen    every candidate applied, tested, reverted exactly, logged as one JSON line
  save      the chosen edit as a NEW checkpoint: refuses to touch the source, verifies the diff, writes a manifest and an
            undo file that rebuilds the original weights
Needs PyTorch 2 (scaled_dot_product_attention); safetensors for .safetensors files and the self-test.
Self-test on a tiny CPU transformer:   python surgery.py --selfcheck
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
import time
import torch


# ------------------------------------------------------------------ the surgeon: reversible edits on a live model
class Surgeon:
    """applies edit specs to a live model and can always put the exact original weights back.
    A spec is a list of ops (dicts, JSON-ready; directions are referred to by name in `dirs`).  The first time a
    parameter is touched its original is copied to the CPU, keyed by parameter name, so any mix of edits reverts
    bit-exactly.  trial(spec) applies, yields, and restores only what that trial changed; apply(spec) keeps the edit."""

    def __init__(self, model, dirs=None):
        self.model, self.dirs = model, dict(dirs or {})
        self.P, self.M = dict(model.named_parameters()), dict(model.named_modules())
        self.pristine, self.kept, self.hooks, self.log = {}, set(), [], []
        self._trial = None

    def _touch(self, name):
        if name not in self.P:
            raise KeyError(f"no parameter {name}")
        if name not in self.pristine:
            self.pristine[name] = self.P[name].detach().cpu().clone()
        if self._trial is not None:
            self._trial.setdefault(name, self.P[name].detach().clone())
        else:
            self.kept.add(name)
        return self.P[name]

    def _dir(self, d):
        return (self.dirs[d] if isinstance(d, str) else torch.as_tensor(d)).float()

    def apply(self, spec):
        for op in spec:
            OPS[op["op"]](self, op)
        if self._trial is None:
            self.log.append(spec)
        return self

    @contextlib.contextmanager
    def trial(self, spec):
        if self._trial is not None:
            raise RuntimeError("trials do not nest")
        self._trial, n_hooks = {}, len(self.hooks)
        try:
            self.apply(spec)
            yield self
        finally:
            for h in self.hooks[n_hooks:]:
                h.remove()
            del self.hooks[n_hooks:]
            with torch.no_grad():
                for n, w in self._trial.items():
                    self.P[n].copy_(w)
            self._trial = None

    def revert(self):
        """everything back to the original weights (kept edits too); hooks removed"""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        with torch.no_grad():
            for n, w in self.pristine.items():
                self.P[n].copy_(w.to(self.P[n].device, self.P[n].dtype))
        self.kept.clear()
        self.log.clear()

    def check_restored(self):
        """every parameter touched only by trials is bit-identical to its original"""
        bad = [n for n, w in self.pristine.items() if n not in self.kept and not torch.equal(self.P[n].detach().cpu(), w)]
        if bad:
            raise AssertionError(f"weights not restored: {bad}")

    def changed(self):
        """{parameter: number of values that differ from the original}"""
        out = {}
        for n, w in self.pristine.items():
            c = int((self.P[n].detach().cpu() != w).sum())
            if c:
                out[n] = c
        return out


def _mat(p, op):
    """the [out, in] view of a weight (layout "in_out" for Conv1D-style weights stored [in, out])"""
    return p.T if op.get("layout") == "in_out" else p


def _index(op, key, n):
    if key in op:
        a, b = op[key]
        return torch.arange(a, b)
    if key + "_idx" in op:
        return torch.as_tensor(op[key + "_idx"], dtype=torch.long)
    return torch.arange(n)


def orthonormal(V, k=None):
    """an orthonormal basis (QR) of the first k columns of V (a [d] vector counts as one column)"""
    V = V.float()
    V = V[:, None] if V.dim() == 1 else V
    Q, _ = torch.linalg.qr(V[:, :k] if k else V)
    return Q


def _op_scale(S, op):
    """multiply part of a weight by `factor`: a head (its input columns of the output projection: cols), a feed-forward
    unit (its column of the down projection: col_idx), a write slot (its row: rows / rows_idx), or the whole tensor.
    factor 0 switches it off; 0.5 halves it; 1.25 boosts it."""
    p = S._touch(op["param"])
    W = _mat(p, op)
    f = float(op["factor"])
    with torch.no_grad():
        if any(k in op for k in ("cols", "cols_idx")):
            W[:, _index(op, "cols", W.shape[1])] *= f
        elif any(k in op for k in ("rows", "rows_idx")):
            W[_index(op, "rows", W.shape[0])] *= f
        else:
            W *= f


def _op_project_in(S, op):
    """inside a head (or any input columns `cols`): cut (scale 0) or shrink the first k directions of dirs[name] out of
    what those columns read: W[:, cols] <- W[:, cols] (I - (1 - scale) Q Q^T), Q orthonormal (QR).  The QR step matters:
    contrast directions are not orthogonal, and I - U U^T on raw unit directions is not a projection for k >= 2."""
    p = S._touch(op["param"])
    W = _mat(p, op)
    c = _index(op, "cols", W.shape[1])
    Q = orthonormal(S._dir(op["dirs"]), op.get("k")).to(W.device)
    with torch.no_grad():
        X = W[:, c].float()
        W[:, c] = (X - (1.0 - float(op.get("scale", 0.0))) * (X @ Q) @ Q.T).to(W.dtype)


def _op_project_out(S, op):
    """remove (strength 1), shrink (<1) or reverse (>1) what a write matrix writes along dirs[name] (its OUTPUT space:
    the residual stream or a side channel), for all input units or only `cols` / `cols_idx`:
        W[:, units] <- W[:, units] - s Q (Q^T W[:, units])
    k=1 with cols_idx = the top writers is the rank-1 edit (s = alpha; alpha 6 makes those units write -5x along u).
    control_seed: the same edit along a random direction orthogonal to Q, rescaled to the same size (the control run)."""
    p = S._touch(op["param"])
    W = _mat(p, op)
    c = _index(op, "cols", W.shape[1])
    Q = orthonormal(S._dir(op["dirs"]), op.get("k", 1)).to(W.device)
    with torch.no_grad():
        X = W[:, c].float()
        delta = -float(op.get("strength", 1.0)) * Q @ (Q.T @ X)
        if op.get("control_seed") is not None:
            g = torch.Generator().manual_seed(int(op["control_seed"]))
            r = torch.randn(Q.shape[0], 1, generator=g).to(Q)
            r = r - Q @ (Q.T @ r)
            r = r / r.norm()
            dr = -(r @ (r.T @ X))
            delta = dr * (delta.norm() / dr.norm().clamp_min(1e-12))
        W[:, c] = (X + delta).to(W.dtype)


def _op_hook(S, op):
    """runtime (activation) edit at a module's output, element `index` if it returns a tuple:
       kind "remove"       x <- x - s (x.d) d           (remove a direction from the stream)
       kind "add"          x <- x + s d                 (steering)
       kind "scale_units"  x[..., units] <- x[..., units] * s + add   (a slot or channel)
    Reversible by removing the hook, but not saveable: bake a "remove" into the module's write matrices with project_out.
    Lesson: removing a direction from the STREAM broke the model; removing it from the WRITING weights did not."""
    mod, kind, idx = S.M[op["module"]], op["kind"], op.get("index", 0)
    s = float(op.get("strength", 1.0))
    d = None if kind == "scale_units" else S._dir(op["dirs"]).flatten()
    d = None if d is None else d / d.norm().clamp_min(1e-12)

    def fn(m, args, out):
        x = out[idx] if isinstance(out, tuple) else out
        if x is None:
            return out
        if kind == "remove":
            dd = d.to(x)
            x = x - s * (x @ dd)[..., None] * dd
        elif kind == "add":
            x = x + s * d.to(x)
        else:
            x = x.clone()
            x[..., op["units"]] = x[..., op["units"]] * s + float(op.get("add", 0.0))
        return out[:idx] + (x,) + out[idx + 1:] if isinstance(out, tuple) else x
    S.hooks.append(mod.register_forward_hook(fn))


OPS = {"scale": _op_scale, "project_in": _op_project_in, "project_out": _op_project_out, "hook": _op_hook}


# ------------------------------------------------------------------ capture: what a module reads or writes
def _run_capture(model, module, batches, forward, select, pre, index):
    got, out = [], []

    def grab(x):
        got.append((x[index] if isinstance(x, tuple) else x).detach())
    h = module.register_forward_pre_hook(lambda m, a: grab(a[0])) if pre else module.register_forward_hook(lambda m, a, o: grab(o))
    try:
        with torch.no_grad():
            for b in batches:
                got.clear()
                forward(model, b)
                X = got[0] if len(got) == 1 else torch.cat(got, dim=-2)   # a chunked forward calls the module once per chunk
                bi, ti = select(b)
                out.append(X[bi, ti].float().cpu())
    finally:
        h.remove()
    return torch.cat(out)


def capture_inputs(model, module, batches, forward, select):
    """what ENTERS `module` (for an attention output projection: all heads' outputs side by side, head h = columns
    h*d:(h+1)*d) at select(batch) -> (batch indices, position indices).  [N, d_in] on the CPU"""
    return _run_capture(model, module, batches, forward, select, True, 0)


def capture_outputs(model, module, batches, forward, select, index=0):
    """what `module` RETURNS (element `index` of a tuple) at the selected positions.  [N, d_out] on the CPU"""
    return _run_capture(model, module, batches, forward, select, False, index)


def decision_positions(prompt_len, n_answer):
    """the position whose OUTPUT predicts answer token j is prompt_len - 1 + j.  Capture there for "what the model writes
    when it produces token j"; prompt_len + j is one step late (a common bug that blurs every contrast)."""
    return [prompt_len - 1 + j for j in range(n_answer)]


# ------------------------------------------------------------------ directions
def mean_direction(X_bad, X_good):
    """unit (mean unwanted - mean wanted).  Use matched rows (same prefix, two continuations): plain onset-minus-start
    averages mix "late in the answer" with "repeating" """
    d = X_bad.float().mean(0) - X_good.float().mean(0)
    return d / d.norm().clamp_min(1e-12)


def writer_attribution(u, W, act_bad, act_good, top=64, layout="out_in"):
    """which input units of write matrix W push the stream along u in the unwanted case: writer = u^T W (each unit's
    write along u), dact = the unit's activation contrast, attr = dact * writer, top = the largest signed attr.
    act_* must be W's OWN input (a mask built from another matrix's units picks the wrong neurons).
    -> (top indices, attr, writer)"""
    Wm = (W.T if layout == "in_out" else W).float()
    writer = u.float() @ Wm
    attr = (act_bad.float().mean(0) - act_good.float().mean(0)) * writer
    return torch.topk(attr, top).indices, attr, writer


def contrast_directions(X_bad, X_good, ridge=1e-3):
    """directions carrying the unwanted job's energy and little of the wanted job's: generalised eigenvectors of
    C_bad v = lam C_good v (uncentred second moments = what the module writes), biggest ratio first, unit length;
    plus each direction's share of both energies and cos(mean bad, mean good) (near 1 = the two jobs share the module).
    The directions are NOT orthogonal: use orthonormal(V, k) before any projection."""
    Cb = X_bad.T.double() @ X_bad.double() / len(X_bad)
    Cg = X_good.T.double() @ X_good.double() / len(X_good)
    Cg = Cg + ridge * torch.trace(Cg) / Cg.shape[0] * torch.eye(Cg.shape[0], dtype=Cg.dtype)
    ev, E = torch.linalg.eigh(Cg)
    Wn = E @ torch.diag(ev.clamp_min(1e-12) ** -0.5) @ E.T
    lam, Z = torch.linalg.eigh(Wn @ Cb @ Wn)
    order = torch.argsort(lam, descending=True)
    V = (Wn @ Z[:, order]).float()
    V = V / V.norm(dim=0, keepdim=True)
    cos = float(torch.nn.functional.cosine_similarity(X_bad.float().mean(0), X_good.float().mean(0), dim=0))
    return V, lam[order].float(), shares(X_bad, V), shares(X_good, V), cos


def shares(X, V):
    """each column's share of X's energy, sum((X v)^2) / sum(X^2); cumulative shares add up only for orthonormal V"""
    return ((X.float() @ V.float()) ** 2).sum(0) / (X.float() ** 2).sum().clamp_min(1e-12)


# ------------------------------------------------------------------ attention: capture and intervene
class AttnTap:
    """patch scaled_dot_product_attention: record each call's attention probabilities and optionally intervene.
       keep(call, q) -> bool   which calls to record / touch (default all; calls are counted in order)
       mask_keys               key indices set to -inf (a sink or a fact removed)
       force                   {query position: key index}: the output there becomes v[key] (a forced one-hot look:
                               separates "where it looks" from "what the rest of the network does with it")
       centre_q                subtract the causal running mean of the queries (a near-constant query shared by all)
    Handles is_causal, boolean or additive masks and grouped KV heads.  `modules`: extra modules that imported
    scaled_dot_product_attention by name.  Index masks are valid only while the key order is unchanged."""

    def __init__(self, keep=None, mask_keys=None, force=None, centre_q=False, store=True, modules=()):
        self.keep, self.mask_keys, self.force, self.centre_q = keep or (lambda i, q: True), mask_keys, force, centre_q
        self.store, self.modules, self.maps, self.calls = store, modules, [], 0

    def __enter__(self):
        F = torch.nn.functional
        self._orig, self._mods = F.scaled_dot_product_attention, []
        tap = self

        def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
            i = tap.calls
            tap.calls += 1
            if not tap.keep(i, q):
                return tap._orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)
            if k.shape[-3] != q.shape[-3]:
                rep = q.shape[-3] // k.shape[-3]
                k, v = k.repeat_interleave(rep, dim=-3), v.repeat_interleave(rep, dim=-3)
            Tq, Tk = q.shape[-2], k.shape[-2]
            qf = q.float()
            if tap.centre_q:
                qf = qf - qf.cumsum(-2) / torch.arange(1, Tq + 1, device=q.device, dtype=qf.dtype)[:, None]
            s = (qf @ k.float().transpose(-1, -2)) * (scale if scale is not None else 1.0 / math.sqrt(q.shape[-1]))
            if is_causal:
                s = s.masked_fill(~torch.ones(Tq, Tk, dtype=torch.bool, device=q.device).tril(Tk - Tq), float("-inf"))
            if attn_mask is not None:
                s = s.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else s + attn_mask.float()
            if tap.mask_keys:
                s[..., list(tap.mask_keys)] = float("-inf")
            a = torch.softmax(s, -1)
            if tap.store:
                tap.maps.append((i, a.detach().cpu()))
            out = (a @ v.float()).to(v.dtype)
            if tap.force:
                out = out.clone()
                for pos, key in tap.force.items():
                    out[..., pos, :] = v[..., key, :]
            return out
        F.scaled_dot_product_attention = sdpa
        for m in self.modules:
            if hasattr(m, "scaled_dot_product_attention"):
                self._mods.append((m, m.scaled_dot_product_attention))
                m.scaled_dot_product_attention = sdpa
        return self

    def __exit__(self, *exc):
        torch.nn.functional.scaled_dot_product_attention = self._orig
        for m, f in self._mods:
            m.scaled_dot_product_attention = f
        return False


def induction_score(attn, ids, offset=1):
    """copy-head score for one sequence, attn [heads, T, T] (query x key).  For every token seen before, the attention it
    pays to its previous occurrence + offset: offset 1 = induction (looks at what came next last time, to copy it),
    offset 0 = plain copy (looks at the previous occurrence itself).  Mean per head"""
    last, hits = {}, []
    for t, x in enumerate(ids):
        if x in last and last[x] + offset < t:
            hits.append((t, last[x] + offset))
        last[x] = t
    if not hits:
        return torch.zeros(attn.shape[0])
    q, k = zip(*hits)
    return attn[:, list(q), list(k)].mean(1)


def induction_scan(model, forward, seqs, offset=1):
    """per attention call (in call order; one call per layer per forward = the layer index) and head: the mean induction
    (offset 1) or copy (offset 0) score over repetitive sequences.  forward(model, ids) runs one sequence."""
    tot = {}
    for ids in seqs:
        with AttnTap() as tap, torch.no_grad():
            forward(model, ids)
        for i, a in tap.maps:
            tot[i] = tot.get(i, 0) + induction_score(a[0], list(ids), offset) / len(seqs)
    return tot


# ------------------------------------------------------------------ probes at fixed decision points (teacher-forced), counts
def loop_onset(ids, n=4, repeats=3):
    """where a loop starts: the index of the SECOND occurrence of the first n-gram that occurs >= `repeats` times; None if none"""
    counts = {}
    for i in range(len(ids) - n + 1):
        g = tuple(ids[i:i + n])
        counts[g] = counts.get(g, 0) + 1
    seen = set()
    for i in range(len(ids) - n + 1):
        g = tuple(ids[i:i + n])
        if counts[g] >= repeats:
            if g in seen:
                return i
            seen.add(g)
    return None


def point_probs(model, logprobs_fn, points):
    """next-token probability of named candidates at fixed decision points.  logprobs_fn(model, ids) -> [len(ids), vocab]
    log-probs; points: [(kind, ids, pos, {candidate: token id})], pos = the index whose output makes the decision.
    -> {kind: {candidate: mean probability, "first is top": share where the first candidate is the top choice}}"""
    acc = {}
    with torch.no_grad():
        for kind, ids, pos, cands in points:
            lp = logprobs_fn(model, ids)[pos]
            a = acc.setdefault(kind, {"n": 0, "first is top": 0.0, **{c: 0.0 for c in cands}})
            a["n"] += 1
            for c, t in cands.items():
                a[c] += float(lp[t].exp())
            a["first is top"] += float(int(lp.argmax()) == next(iter(cands.values())))
    return {k: {c: v / a["n"] for c, v in a.items() if c != "n"} for k, a in acc.items()}


def onset_probe(model, logprobs_fn, samples):
    """at real loop onsets: P(the repeated token) and how often it is the top choice.  samples: [(ids, onset index)]"""
    r = point_probs(model, logprobs_fn, [("onset", ids[:o], o - 1, {"repeat": ids[o]}) for ids, o in samples])["onset"]
    return {"P(repeat)": r["repeat"], "repeat is top": r["first is top"]}


def tail_probe(model, logprobs_fn, pairs):
    """matched continuations of one prefix: mean log-prob per token of the productive and of the repetitive branch, and
    the log-prob of the close's first token.  pairs: [(head_ids, productive_ids, repetitive_ids, close_ids)].
    A good edit: repetitive drops a lot, productive barely moves, close stays put."""
    acc = {"productive": 0.0, "repetitive": 0.0, "close": 0.0}
    with torch.no_grad():
        for head, prod, rep, close in pairs:
            for key, br in (("productive", prod), ("repetitive", rep)):
                ids = list(head) + list(br)
                lp = logprobs_fn(model, ids[:-1])
                tok = lp[torch.arange(len(ids) - 1), torch.tensor(ids[1:])]
                acc[key] += float(tok[len(head) - 1:].mean())
            acc["close"] += float(logprobs_fn(model, list(head))[len(head) - 1, close[0]])
    return {k: v / len(pairs) for k, v in acc.items()}


def rate_change(k1, n1, k2, n2):
    """is a change in a counted rate real?  two-proportion z, |z| >= 2 ~ real (6/200 -> 9/200 is noise, 17/200 -> 40/200 is not)"""
    p1, p2, p = k1 / n1, k2 / n2, (k1 + k2) / (n1 + n2)
    se = math.sqrt(max(p * (1 - p) * (1 / n1 + 1 / n2), 1e-30))
    return dict(change=p2 - p1, se=se, z=(p2 - p1) / se, real=abs(p2 - p1) / se >= 2)


def exchange(results, cost, benefit, sign=-1.0, noise=0.0):
    """cost paid per unit of benefit for every row of {row: {metric: change}}, best deal first.  benefit = sign * change
    (sign -1 when the wanted change is a drop, e.g. P(repeat)).  Rows whose benefit is under `noise` are marked noisy."""
    out = []
    for r, m in results.items():
        b, c = sign * m[benefit], m[cost]
        out.append(dict(row=r, cost=c, benefit=b, per_unit=c / b if b > 0 else float("inf"), noisy=abs(b) < noise))
    return sorted(out, key=lambda x: x["per_unit"])


def table(results, metrics=None, fmt="{:+.4f}"):
    """{row: {metric: value}} as a markdown table"""
    rows_ = list(results)
    ms = metrics or list(results[rows_[0]])
    lines = ["| | " + " | ".join(ms) + " |", "|---" * (len(ms) + 1) + "|"]
    lines += ["| " + str(r) + " | " + " | ".join(fmt.format(results[r][m]) for m in ms) + " |" for r in rows_]
    return "\n".join(lines)


# ------------------------------------------------------------------ scans and screens
def head_scan(model, heads, tests, out_proj, head_dim, scales=(0.0,), layout="out_in"):
    """every head switched off (scale 0) or scaled, one at a time, against the baseline: {tag: change per metric}.
    heads: [(layer, head)]; out_proj(layer) -> the output projection's parameter name (head h = its input columns).
    "Responsible" means the metric collapses without it; a head that every scale makes worse is a foundation, not harm."""
    S = Surgeon(model)
    base = tests(model)
    out = {}
    for layer, h in heads:
        for s in scales:
            spec = [dict(op="scale", param=out_proj(layer), cols=[h * head_dim, (h + 1) * head_dim], factor=s, layout=layout)]
            with S.trial(spec):
                m = tests(model)
            out[f"layer {layer} head {h} x{s:g}"] = {x: m[x] - base[x] for x in m}
    S.check_restored()
    return base, out


def screen(model, candidates, tests, dirs=None, log=None):
    """every candidate spec applied, tested, reverted (checked bit-exact), and logged: {tag: change per metric}.
    candidates: {tag: spec}; tests(model) -> {metric: value} on FIXED rows; log: a JSONL path (one line per candidate)."""
    S = Surgeon(model, dirs)
    base = tests(model)
    rows = {}
    for tag, spec in candidates.items():
        t0 = time.time()
        with S.trial(spec):
            m = tests(model)
        S.check_restored()
        rows[tag] = {x: m[x] - base[x] for x in m}
        if log:
            with open(log, "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(tag=tag, spec=spec, seconds=round(time.time() - t0, 2), base=base, change=rows[tag])) + "\n")
    return base, rows


def dose_series(spec_fn, doses):
    """{f"x{d}": spec_fn(d)} for a dose series (alpha 1, 2, 4, 6 ...): a real effect grows with the dose"""
    return {f"x{d:g}": spec_fn(d) for d in doses}


# ------------------------------------------------------------------ saving the edited checkpoint
def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def tensor_sha256(t):
    return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def _load(path, state_key):
    """(payload, state dict): a torch.save payload with the weights under state_key (state_key None: the file IS the
    state dict), or a .safetensors file (a Hugging Face single-file checkpoint; payload None)"""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return None, load_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload, (payload[state_key] if state_key else payload)


def _store(path, payload, sd, state_key, meta):
    if path.endswith(".safetensors"):
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in sd.items()}, path, metadata={k: json.dumps(v) for k, v in meta.items()})
    elif state_key:
        payload[state_key] = sd
        payload["extra"] = dict(payload.get("extra") or {}, **meta)
        torch.save(payload, path)
    else:
        torch.save(sd, path)


def save_edited(model, src, out, spec, surgeon=None, state_key="model", note=None):
    """the model's current weights saved as a NEW checkpoint built from the source (a .pt payload: every other key kept,
    provenance in payload["extra"]["weight_surgery"]; or a .safetensors file: provenance in its metadata).  The source's
    tensor set and dtypes are kept.  Refuses to write the source or any existing file and refuses runtime hook edits,
    then verifies: the reloaded file differs from the source in exactly the edited tensors, nothing outside the surgeon's
    touched tensors changed, and the source file's hash is unchanged.  Writes out + ".edit.json" (manifest: per tensor
    shape, dtype, changed entries / rows / columns, relative change, base and edited sha256) and out + ".undo.pt" (the
    original tensors and the directions used).  For a sharded Hugging Face directory use model.save_pretrained(new_dir)
    and diff the two state dicts the same way.  -> the manifest"""
    src, out = os.path.abspath(src), os.path.abspath(out)
    if out == src:
        raise ValueError("never write the source checkpoint")
    if os.path.exists(out):
        raise FileExistsError(f"{out} exists: pick a new name")
    if src.endswith(".safetensors") != out.endswith(".safetensors"):
        raise ValueError("keep the source's file format")
    if surgeon is not None and surgeon.hooks:
        raise ValueError("runtime hook edits can't be saved: bake them into the weights with project_out")
    h0 = file_sha256(src)
    payload, old = _load(src, state_key)
    cur = model.state_dict()
    missing = set(old) - set(cur)
    if missing:
        raise KeyError(f"checkpoint tensors the model does not have: {sorted(missing)[:6]}")
    new = {k: cur[k].detach().cpu().clone().to(old[k].dtype) for k in old}
    changed = [k for k in new if not torch.equal(new[k], old[k])]
    if not changed:
        raise ValueError("nothing changed (an edit smaller than the checkpoint's precision rounds away)")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    _store(out, payload, new, state_key, dict(weight_surgery=dict(spec=spec, source=src, source_sha256=h0, note=note)))
    torch.save(dict(originals={k: old[k].clone() for k in changed}, dirs=dict(surgeon.dirs) if surgeon else {}), out + ".undo.pt")
    _, re_ = _load(out, state_key)
    diff = [k for k in old if not torch.equal(old[k], re_[k])]
    assert sorted(diff) == sorted(changed), "the saved file differs from the intended edit"
    if surgeon is not None:
        stray = set(diff) - set(surgeon.pristine)
        assert not stray, f"tensors changed outside the edit: {sorted(stray)[:6]}"
    assert file_sha256(src) == h0, "the source checkpoint changed!"
    tensors = {}
    for k in diff:
        ne = old[k] != re_[k]
        m2 = ne.reshape(ne.shape[0], -1) if ne.dim() >= 2 else ne.reshape(1, -1)
        tensors[k] = dict(shape=list(old[k].shape), dtype=str(old[k].dtype), changed_entries=int(ne.sum()),
                          changed_rows=int(m2.any(1).sum()), changed_cols=int(m2.any(0).sum()),
                          relative_change=float((re_[k].float() - old[k].float()).norm() / old[k].float().norm().clamp_min(1e-12)),
                          base_sha256=tensor_sha256(old[k]), edited_sha256=tensor_sha256(re_[k]))
    manifest = dict(source=src, source_sha256=h0, out=out, out_sha256=file_sha256(out), spec=spec, note=note,
                    changed_tensors=len(diff), changed_entries_total=sum(t["changed_entries"] for t in tensors.values()), tensors=tensors)
    with open(out + ".edit.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def undo_saved(edited, out, state_key="model", source=None):
    """rebuild the original weights from an edited checkpoint and its .undo.pt, as a NEW file in the same format; with
    `source`, verify the rebuilt weights equal the source's bit for bit"""
    edited, out = os.path.abspath(edited), os.path.abspath(out)
    if out == edited or os.path.exists(out):
        raise FileExistsError(f"{out}: pick a new name")
    payload, sd = _load(edited, state_key)
    for k, v in torch.load(edited + ".undo.pt", map_location="cpu", weights_only=False)["originals"].items():
        sd[k] = v
    _store(out, payload, sd, state_key, dict(weight_surgery_undone=edited))
    if source:
        _, s0 = _load(source, state_key)
        bad = [k for k in s0 if not torch.equal(s0[k], sd[k])]
        assert not bad, f"undo did not rebuild the source: {bad[:6]}"
    return out


# ------------------------------------------------------------------ self-check (tiny CPU transformer, seconds)
class _Block(torch.nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = torch.nn.LayerNorm(d), torch.nn.LayerNorm(d)
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = (torch.nn.Linear(d, d, bias=False) for _ in range(4))
        self.up_proj, self.down_proj = torch.nn.Linear(d, 4 * d), torch.nn.Linear(4 * d, d)

    def forward(self, x):
        B, T, D = x.shape
        y = self.ln1(x)
        q, k, v = (f(y).view(B, T, self.h, -1).transpose(1, 2) for f in (self.q_proj, self.k_proj, self.v_proj))
        a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o_proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.down_proj(torch.nn.functional.gelu(self.up_proj(self.ln2(x))))


class _Tiny(torch.nn.Module):
    def __init__(self, V=40, d=32, h=4, L=3, T=24):
        super().__init__()
        self.embed = torch.nn.Embedding(V, d)
        self.pos = torch.nn.Parameter(torch.randn(T, d) * 0.02)
        self.layers = torch.nn.ModuleList(_Block(d, h) for _ in range(L))
        self.norm = torch.nn.LayerNorm(d)
        self.lm_head = torch.nn.Linear(d, V, bias=False)

    def forward(self, ids):
        x = self.embed(ids) + self.pos[: ids.shape[1]]
        for b in self.layers:
            x = b(x)
        return self.lm_head(self.norm(x))


def _copy_batch(seed, B=32, T=24, V=40):
    g = torch.Generator().manual_seed(seed)
    first = torch.randint(2, V, (B, T // 2), generator=g)
    return torch.cat([first, first], 1)                   # the second half repeats the first: a copy (induction) task


def _trained_tiny():
    torch.manual_seed(0)
    F = torch.nn.functional
    model = _Tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for s in range(1000):                                 # ~800 steps: it learns to copy and grows an induction head
        x = _copy_batch(10_000 + s)
        loss = F.cross_entropy(model(x[:, :-1]).reshape(-1, 40), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


def selfcheck():
    import tempfile
    F = torch.nn.functional
    model = _trained_tiny()
    held = _copy_batch(999)

    def tests(m):
        with torch.no_grad():
            lg = m(held[:, :-1])[:, 11:]
            return {"copy loss": float(F.cross_entropy(lg.reshape(-1, 40), held[:, 12:].reshape(-1))),
                    "copy top-1": float((lg.argmax(-1) == held[:, 12:]).float().mean())}
    base = tests(model)
    W0 = {n: p.detach().clone() for n, p in model.named_parameters()}
    O = "layers.1.o_proj.weight"

    # 1. scale a head, trial on top of a kept edit, exact revert
    S = Surgeon(model)
    S.apply([dict(op="scale", param=O, cols=[0, 8], factor=0.0)])
    assert S.changed() == {O: 32 * 8}, S.changed()
    with S.trial([dict(op="scale", param="layers.0.up_proj.weight", rows=[0, 4], factor=2.0)]):
        assert "layers.0.up_proj.weight" in S.changed()
    assert set(S.changed()) == {O}, S.changed()                          # the trial is gone, the kept edit stays
    S.check_restored()
    S.revert()
    assert all(torch.equal(W0[n], p) for n, p in model.named_parameters()), "revert is not exact"
    print(f"[ok] scale, trial over a kept edit, exact revert (copy top-1 {base['copy top-1']:.2f})")

    # 2. project_in cuts exactly the span of non-orthogonal directions; the old I - U U^T did not
    U = torch.nn.functional.normalize(torch.randn(8, 3) + 2.0, dim=0)  # unit, far from orthogonal
    S = Surgeon(model, {"u": U})
    S.apply([dict(op="project_in", param=O, cols=[8, 16], dirs="u", k=3)])
    Q = orthonormal(U, 3)
    assert float((model.layers[1].o_proj.weight[:, 8:16].detach() @ Q).abs().max()) < 1e-5
    old_bug = W0[O][:, 8:16] @ (torch.eye(8) - U @ U.T)
    assert float((old_bug @ Q).abs().max()) > 1e-2, "the oblique form should leave a residue"
    S.revert()
    print("[ok] project_in removes the whole span (the unorthogonalised form leaves a residue)")

    # 3. project_out: the rank-1 edit on chosen units, and its norm-matched random control
    D = "layers.1.down_proj.weight"
    u = torch.nn.functional.normalize(torch.randn(32), dim=0)
    idx = [3, 17, 40, 99]
    S = Surgeon(model, {"u": u})
    S.apply([dict(op="project_out", param=D, dirs="u", strength=1.0, cols_idx=idx)])
    Wd = model.layers[1].down_proj.weight.detach()
    assert float((u @ Wd[:, idx]).abs().max()) < 1e-5 and S.changed()[D] <= 32 * len(idx)
    real = Wd - W0[D]
    S.revert()
    S.apply([dict(op="project_out", param=D, dirs="u", strength=1.0, cols_idx=idx, control_seed=7)])
    ctrl = model.layers[1].down_proj.weight.detach() - W0[D]
    assert abs(float(ctrl.norm() - real.norm())) < 1e-4 * float(real.norm()) and float((u @ ctrl).abs().max()) < 1e-5
    S.revert()
    print("[ok] project_out (rank-1 on chosen units) and its norm-matched orthogonal control")

    # 4. runtime hook: the direction is gone from the block output while hooked; outputs identical after
    d = torch.nn.functional.normalize(torch.randn(32), dim=0)

    def block0_out():                                                    # an observer registered last sees the edited output
        got = []
        h = model.layers[0].register_forward_hook(lambda m, a, o: got.append(o.detach()))
        with torch.no_grad():
            model(held[:2, :-1])
        h.remove()
        return got[0]
    before = block0_out()
    S = Surgeon(model, {"d": d})
    with S.trial([dict(op="hook", module="layers.0", kind="remove", dirs="d")]):
        during = block0_out()
    after = block0_out()
    assert float((during @ d).abs().max()) < 1e-4 and torch.equal(before, after) and not S.hooks
    print("[ok] runtime hook edit, removed cleanly")

    # 5. directions: contrast recovers a planted direction; writer attribution finds the planted unit
    g = torch.Generator().manual_seed(1)
    Xg = torch.randn(4000, 8, generator=g)
    Xb = torch.randn(4000, 8, generator=g)
    Xb[:, 3] += 3 * torch.randn(4000, generator=g)
    V, lam, sb, sg, cos = contrast_directions(Xb, Xg)
    assert abs(float(V[3, 0])) > 0.9 and float(sb[0]) > float(sg[0]) and float(lam[0]) > float(lam[-1])
    Qo = orthonormal(V, 4)
    assert torch.allclose(Qo.T @ Qo, torch.eye(4), atol=1e-5)
    Wt = torch.randn(32, 128, generator=g) * 0.1
    Wt[:, 50] += 2.0 * u
    a_good = torch.rand(200, 128, generator=g)
    a_bad = a_good.clone()
    a_bad[:, 50] += 1.0
    top, attr, writer = writer_attribution(u, Wt, a_bad, a_good, top=3)
    assert int(top[0]) == 50, top
    print("[ok] contrast directions (planted one found, orthonormalised) and writer attribution")

    # 6. attention: tap rows sum to 1 and are causal; the copy model has an induction head; interventions work
    seqs = [_copy_batch(5000 + i, B=1)[0].tolist() for i in range(4)]

    def fwd(m, ids):
        return m(torch.tensor([ids]))
    scores = induction_scan(model, fwd, seqs)
    best = max(float(s.max()) for s in scores.values())
    med = float(torch.cat([s for s in scores.values()]).median())
    with AttnTap() as tap, torch.no_grad():
        fwd(model, seqs[0])
    a = tap.maps[0][1][0]
    assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-4) and float(a.triu(1).abs().max()) == 0.0
    assert best > 0.3 and best > 3 * med, (best, med)
    with AttnTap(keep=lambda i, q: i == 0, mask_keys=[0]) as tap, torch.no_grad():
        fwd(model, seqs[0])
    assert float(tap.maps[0][1][0][:, 1:, 0].abs().max()) == 0.0
    with torch.no_grad():
        plain = fwd(model, seqs[0])
        with AttnTap(keep=lambda i, q: i == 0, force={20: 3}):
            forced = fwd(model, seqs[0])
    assert not torch.allclose(plain, forced)
    print(f"[ok] attention tap, induction scan (best head {best:.2f} vs median {med:.2f}), key mask, forced look")

    # 7. head scan and screen leave the weights exact; the screen log has one line per candidate
    _, hs = head_scan(model, [(l, h) for l in range(3) for h in range(4)], tests, lambda l: f"layers.{l}.o_proj.weight", 8)
    assert len(hs) == 12 and all(torch.equal(W0[n], p) for n, p in model.named_parameters())
    worst = max(hs, key=lambda t: hs[t]["copy loss"])
    with tempfile.TemporaryDirectory() as tmp:
        logf = os.path.join(tmp, "screen.jsonl")
        _, sc = screen(model, {"cut 2": [dict(op="project_in", param=O, cols=[8, 16], dirs="u", k=2)],
                               "doses": [dict(op="project_out", param=D, dirs="w", strength=2.0)]},
                       tests, dirs={"u": U, "w": u}, log=logf)
        assert len(open(logf, encoding="utf-8").readlines()) == 2
        assert all(torch.equal(W0[n], p) for n, p in model.named_parameters())
        ranked = exchange(sc, cost="copy loss", benefit="copy top-1", sign=1.0)
        assert len(ranked) == 2 and table(sc).count("\n") == 3
        print(f"[ok] head scan (most needed: {worst}, copy loss {hs[worst]['copy loss']:+.3f}) and screen, weights exact")

        # 8. save as a new checkpoint, verify, refuse unsafe writes, undo to the source bit for bit
        src = os.path.join(tmp, "base.pt")
        torch.save(dict(model=model.state_dict(), step=7, extra=dict(run="selfcheck")), src)
        from safetensors.torch import save_file
        st_src = os.path.join(tmp, "base.safetensors")
        save_file({k: v.contiguous() for k, v in model.state_dict().items()}, st_src)
        S = Surgeon(model, {"u": U})
        spec = [dict(op="project_in", param=O, cols=[8, 16], dirs="u", k=2)]
        S.apply(spec)
        try:
            save_edited(model, src, src, spec, S)
            raise AssertionError("wrote over the source")
        except ValueError:
            pass
        out = os.path.join(tmp, "edits", "base_cut2.pt")
        man = save_edited(model, src, out, spec, S, note="selfcheck")
        try:
            save_edited(model, src, out, spec, S)
            raise AssertionError("overwrote an existing file")
        except FileExistsError:
            pass
        assert man["changed_tensors"] == 1 and O in man["tensors"] and man["tensors"][O]["changed_cols"] <= 8
        assert torch.load(out, weights_only=False)["extra"]["weight_surgery"]["spec"] == spec
        back = undo_saved(out, os.path.join(tmp, "base_rebuilt.pt"), source=src)
        assert os.path.exists(back)
        st_out = os.path.join(tmp, "edits", "base_cut2.safetensors")
        st_man = save_edited(model, st_src, st_out, spec, S, state_key=None)
        assert st_man["changed_tensors"] == 1 and st_man["changed_entries_total"] == man["changed_entries_total"]
        undo_saved(st_out, os.path.join(tmp, "base_rebuilt.safetensors"), state_key=None, source=st_src)
        S.revert()
        print(f"[ok] saved a new checkpoint ({man['changed_entries_total']} numbers in {man['changed_tensors']} tensor), "
              "verified (.pt and .safetensors), unsafe writes refused, undo rebuilt the source exactly")

    # 9. probes and counts
    def lp_fn(m, ids_):
        return torch.log_softmax(m(torch.tensor([ids_]))[0], -1)
    pp = tail_probe(model, lp_fn, [(seqs[0][:12], seqs[0][12:18], seqs[0][6:12], [seqs[0][12]])])
    assert all(math.isfinite(v) for v in pp.values())
    assert loop_onset([1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6]) == 8 and loop_onset([1, 2, 3, 4, 5]) is None
    assert not rate_change(6, 200, 9, 200)["real"] and rate_change(17, 200, 40, 200)["real"]
    seq = held[0].tolist()
    pr = point_probs(model, lp_fn, [("copy", seq[:20], 19, {"gold": seq[20], "other": (seq[20] + 1) % 40})])["copy"]
    assert 0.0 <= pr["other"] <= 1.0 and pr["gold"] > pr["other"], pr
    on = onset_probe(model, lp_fn, [(seq, 13)])
    assert 0.0 <= on["P(repeat)"] <= 1.0 and on["repeat is top"] in (0.0, 1.0)
    print(f"[ok] tail probe, loop onset, count noise, point and onset probes (gold {pr['gold']:.2f})")
    print("selfcheck passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selfcheck", action="store_true", help="run every tool on a tiny CPU transformer")
    if ap.parse_args().selfcheck:
        selfcheck()
    else:
        ap.print_help()
