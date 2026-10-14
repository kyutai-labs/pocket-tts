"""Build a smaller checkpoint from a bigger one: fewer layers, and optionally narrower.

Depth: every tensor outside the backbone is copied unchanged; the backbone keeps
the teacher's bottom and top layers (see select_layers). Use it to initialize a
depth-distillation student -- starting from the teacher's own layers converges
far faster than a random backbone -- or to get a standalone shallow checkpoint.

    python -m training.scripts.shrink_checkpoint runs/teacher/checkpoint_00200000.pt \
        --layers 6 --out runs/student_init.pt

Width: with --config (a pocket-tts model config, edited by --override) the flow LM
is seeded for that architecture instead, see shrink_width: its depth, a narrower
d_model, fewer or grouped attention heads (attn_dim, num_kv_heads), a narrower FFN
and a low-rank text table. With --cov (from training.scripts.residual_cov), d_model
is cut along the principal components of the residual stream instead (pca_channels).
Tensors that cannot be seeded (e.g. a sampler head of another size) are left out and
listed.

    python -m training.scripts.shrink_checkpoint runs/teacher/checkpoint_00200000.pt \
        --config pocket_tts/config/english.yaml \
        --override flow_lm.transformer.d_model=512 --override flow_lm.lookup_table.dim=512 \
        --out runs/narrow_init.pt
"""

import argparse
import re
from pathlib import Path
from typing import Any

import torch
import yaml

Tensors = dict[str, torch.Tensor]


def select_layers(n_teacher: int, n_student: int) -> list[int]:
    """Which teacher layers the student starts from: the bottom half and the
    top half of the stack ("ends").

    Won the 4-way ablation (ends/last/spaced/first, 2026-08): top-heavy seeds
    reach teacher parity ~20k steps sooner than bottom-heavy ones, and ends
    edged out last at every checkpoint from 30k on.
    """
    assert 0 < n_student <= n_teacher, f"cannot shrink {n_teacher} -> {n_student}"
    head = (n_student + 1) // 2
    tail = n_student - head
    return list(range(head)) + list(range(n_teacher - tail, n_teacher))


def shrink(state: Tensors, n_student: int) -> tuple[Tensors, list[int]]:
    pattern = re.compile(r"^(flow_lm\.)?transformer\.layers\.(\d+)\.(.*)$")
    depth = max((int(m.group(2)) for m in (pattern.match(k) for k in state) if m), default=-1) + 1
    assert depth > 0, "no transformer layers found in the checkpoint"
    keep = select_layers(depth, n_student)
    remap = {old: new for new, old in enumerate(keep)}
    out: Tensors = {}
    for k, v in state.items():
        m = pattern.match(k)
        if not m:
            out[k] = v
            continue
        old = int(m.group(2))
        if old not in remap:
            continue
        out[f"{m.group(1) or ''}transformer.layers.{remap[old]}.{m.group(3)}"] = v
    return out, keep


def layer_prefixes(state: Tensors) -> list[str]:
    """The "transformer.layers.<i>" prefixes of a flow LM state dict (no "flow_lm.")."""
    n = 0
    while f"transformer.layers.{n}.norm1.weight" in state:
        n += 1
    return [f"transformer.layers.{i}" for i in range(n)]


def narrow_attention(src: Tensors, dst: Tensors, head_dim: int) -> Tensors:
    """Fewer query heads and/or grouped kv heads, at the source's d_model.

    Per layer, keeps the query heads with the largest q/k/v weight norm; each kv head
    of the result is the mean of the source kv heads its query heads read. Both sides
    must have the same head width. Rows of in_proj are [q | k | v].
    """
    out: Tensors = {}
    for layer in layer_prefixes(dst):
        k_in, k_out = f"{layer}.self_attn.in_proj.weight", f"{layer}.self_attn.out_proj.weight"
        if k_in not in src or src[k_in].shape[0] == dst[k_in].shape[0]:
            continue
        w, d = src[k_in], src[k_in].shape[1]
        heads_t, heads_s = src[k_out].shape[1] // head_dim, dst[k_out].shape[1] // head_dim
        kv_t = (w.shape[0] - heads_t * head_dim) // (2 * head_dim)
        kv_s = (dst[k_in].shape[0] - heads_s * head_dim) // (2 * head_dim)
        if min(kv_t, kv_s) < 1 or heads_s > heads_t or heads_t % kv_t or heads_s % kv_s:
            raise ValueError(f"{layer}: cannot seed {heads_s}/{kv_s} heads from {heads_t}/{kv_t}")
        q, k, v = w.split([heads_t * head_dim, kv_t * head_dim, kv_t * head_dim])
        q, k, v = (x.reshape(-1, head_dim, d) for x in (q, k, v))
        group = torch.arange(heads_t) // (heads_t // kv_t)  # the kv head each query head reads
        score = q.square().sum((1, 2)) + k[group].square().sum((1, 2))
        score = score + v[group].square().sum((1, 2))
        keep = score.topk(heads_s).indices.sort().values
        k = k[group[keep]].reshape(kv_s, heads_s // kv_s, head_dim, d).mean(1)
        v = v[group[keep]].reshape(kv_s, heads_s // kv_s, head_dim, d).mean(1)
        out[k_in] = torch.cat([q[keep].reshape(-1, d), k.reshape(-1, d), v.reshape(-1, d)])
        o = src[k_out].reshape(src[k_out].shape[0], heads_t, head_dim)[:, keep]
        out[k_out] = o.reshape(o.shape[0], heads_s * head_dim)
    return out


def residual_axes(state: Tensors) -> dict[str, int]:
    """Every tensor with a d_model (residual stream) axis: key -> that axis."""
    axes = {
        "out_norm.weight": 0,
        "out_norm.bias": 0,
        "bos_before_voice": 2,
        "speaker_proj_weight": 0,
        "input_linear.weight": 0,
        "voice_table.embed.weight": 1,
        "conditioner.proj.weight": 0,  # low-rank text table
        # The full text table, unless the low-rank one's projection carries d_model.
        **({} if "conditioner.proj.weight" in state else {"conditioner.embed.weight": 1}),
        # Readers of the backbone output: the z bottleneck, else both heads.
        **(
            {"z_proj.weight": 1}
            if "z_proj.weight" in state
            else {"out_eos.weight": 1, "flow_net.cond_embed.weight": 1}
        ),
    }
    for layer in layer_prefixes(state):
        axes.update(
            {
                f"{layer}.norm1.weight": 0,
                f"{layer}.norm1.bias": 0,
                f"{layer}.norm2.weight": 0,
                f"{layer}.norm2.bias": 0,
                f"{layer}.self_attn.in_proj.weight": 1,
                f"{layer}.self_attn.out_proj.weight": 0,
                f"{layer}.linear1.weight": 1,
                f"{layer}.linear2.weight": 0,
            }
        )
    return {k: a for k, a in axes.items() if k in state}


def slice_channels(src: Tensors, d_student: int) -> Tensors:
    """Narrower d_model: keep the residual channels with the largest LayerNorm gains
    (summed over every norm of the backbone), in their original order."""
    score = src["out_norm.weight"].abs().clone()
    for layer in layer_prefixes(src):
        score += src[f"{layer}.norm1.weight"].abs() + src[f"{layer}.norm2.weight"].abs()
    keep = score.topk(d_student).indices.sort().values
    return {k: src[k].index_select(axis, keep) for k, axis in residual_axes(src).items()}


def pca_channels(src: Tensors, d_student: int, cov: Tensors) -> Tensors:
    """Narrower d_model along the principal components of the residual stream (as in
    SliceGPT) instead of a subset of its channels.

    `cov` holds, per backbone LayerNorm ("transformer.layers.<i>.norm1", ..., "out_norm"),
    the second moment of its normalized input (training/scripts/residual_cov.py). The
    stream is only read through LayerNorms, so its mean direction is invisible: writers
    are centred, then mapped by M^T, with M (d, d_student) orthonormal, spanning the top
    d_student - 1 components plus the mean direction, which lands on the student's own
    mean direction. Norm gains are folded into the linears that read them (the student's
    gains are 1, rescaled for the variance the cut keeps) and each norm's bias is refit
    by least squares on its readers.
    """
    f64 = torch.float64
    d = src["out_norm.weight"].shape[0]
    sigma = sum(c.to(f64) for c in cov.values()) / len(cov)
    center = torch.eye(d, dtype=f64) - 1.0 / d
    evals, evecs = torch.linalg.eigh(center @ sigma @ center)
    top = evecs[:, evals.argsort(descending=True)[: d_student - 1]]
    basis = torch.cat([torch.full((d, 1), d**-0.5, dtype=f64), top], dim=1)
    # Householder reflection taking e_1 to the student's mean direction.
    w = torch.zeros(d_student, dtype=f64)
    w[0] = 1.0
    w = w - d_student**-0.5
    reflect = torch.eye(d_student, dtype=f64) - 2 * torch.outer(w, w) / w.dot(w)
    M = basis @ reflect  # (d, d_student)
    to_student = M.T @ center  # applied to everything written into the stream

    axes = residual_axes(src)
    out: Tensors = {}

    def put(key: str, value: torch.Tensor):
        out[key] = value.to(src[key].dtype)

    norms = [f"{layer}.{n}" for layer in layer_prefixes(src) for n in ("norm1", "norm2")]
    norms.append("out_norm")
    readers = {
        f"{layer}.norm1": [f"{layer}.self_attn.in_proj.weight"] for layer in layer_prefixes(src)
    }
    readers.update({f"{layer}.norm2": [f"{layer}.linear1.weight"] for layer in layer_prefixes(src)})
    readers["out_norm"] = [
        k for k in ("z_proj.weight", "out_eos.weight", "flow_net.cond_embed.weight") if k in axes
    ]
    norm_keys = {f"{n}.{p}" for n in norms for p in ("weight", "bias")}
    read_keys = {k for ks in readers.values() for k in ks}
    for key, axis in axes.items():
        if key in norm_keys or key in read_keys:
            continue
        value = src[key].to(f64)
        # Writers: rows of a linear (axis 0), or the last axis of an embedding-like tensor.
        put(key, to_student @ value if axis == 0 else value @ to_student.T)

    for norm in norms:
        gain, bias = src[f"{norm}.weight"].to(f64), src[f"{norm}.bias"].to(f64)
        c = cov[norm].to(f64)
        kept = torch.trace(M.T @ c @ M) / torch.trace(c)  # share of the variance the cut keeps
        scale = (d * kept / d_student) ** 0.5
        weights, targets = [], []
        for key in readers[norm]:
            reader = src[key].to(f64)
            weights.append((reader * gain) @ M * scale)
            targets.append(reader @ bias)
            put(key, weights[-1])
        fit = torch.linalg.lstsq(torch.cat(weights), torch.cat(targets)[:, None]).solution
        put(f"{norm}.weight", torch.ones(d_student, dtype=f64))
        put(f"{norm}.bias", fit[:, 0])
    return out


def prune_ffn(src: Tensors, dst: Tensors) -> Tensors:
    """Narrower FFN: keep the hidden units with the largest ||linear1 row|| * ||linear2 col||."""
    out: Tensors = {}
    for layer in layer_prefixes(dst):
        k1, k2 = f"{layer}.linear1.weight", f"{layer}.linear2.weight"
        hidden = dst[k1].shape[0]
        if k1 not in src or src[k1].shape[0] <= hidden:
            continue
        l1, l2 = src[k1], src[k2]
        keep = (l1.norm(dim=1) * l2.norm(dim=0)).topk(hidden).indices.sort().values
        out[k1], out[k2] = l1[keep], l2[:, keep]
    return out


def low_rank_text_table(src: Tensors, dst: Tensors) -> Tensors:
    """A low-rank text table from a full one: its best rank-r approximation (SVD),
    E ~= (U_r S_r) @ Vh_r, as conditioner.embed (U_r S_r) and conditioner.proj (Vh_r^T)."""
    if "conditioner.proj.weight" not in dst or "conditioner.proj.weight" in src:
        return {}
    full = src["conditioner.embed.weight"]
    rank = dst["conditioner.embed.weight"].shape[1]
    u, s, vh = torch.linalg.svd(full.float(), full_matrices=False)
    return {
        "conditioner.embed.weight": (u[:, :rank] * s[:rank]).to(full.dtype),
        "conditioner.proj.weight": vh[:rank].T.contiguous().to(full.dtype),
    }


def shrink_width(src: Tensors, dst: Tensors, head_dim: int, cov: Tensors | None = None) -> Tensors:
    """Seed the flow LM state `dst` (only its shapes are read) from `src` of the same depth.

    Attention heads are narrowed first (narrow_attention), then the residual stream
    (slice_channels, or pca_channels given the residual covariance `cov`), the FFN
    (prune_ffn) and the text table (low_rank_text_table). Returns the tensors that end
    up with `dst`'s shapes.
    """
    state = {**src, **narrow_attention(src, dst, head_dim)}
    d_dst = dst["out_norm.weight"].shape[0]
    if state["out_norm.weight"].shape[0] != d_dst:
        if cov is None:
            state.update(slice_channels(state, d_dst))
        else:
            state.update(pca_channels(state, d_dst, cov))
    state.update(prune_ffn(state, dst))
    state.update(low_rank_text_table(state, dst))
    return {k: v for k, v in state.items() if k in dst and dst[k].shape == v.shape}


def flow_lm_shapes(config_path: str, overrides: dict[str, Any]) -> tuple[Tensors, int]:
    """(state dict, attention head width) of the flow LM a pocket-tts config builds."""
    from pocket_tts.models.flow_lm import FlowLMModel
    from training.modules.builders import load_model_config

    config = load_model_config(config_path, overrides)
    fl = config.flow_lm
    flow_lm = FlowLMModel.from_pydantic_config(
        fl, config.mimi.quantizer.dimension, fl.insert_bos_before_voice
    )
    state = flow_lm.state_dict()
    latent_dim = config.mimi.inner_dim or config.mimi.seanet.dimension
    state["speaker_proj_weight"] = torch.empty(fl.transformer.d_model, latent_dim)
    attn_dim = fl.transformer.attn_dim or fl.transformer.d_model
    return state, attn_dim // fl.transformer.num_heads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--layers", type=int, help="student depth (depth-only shrinking)")
    parser.add_argument("--config", help="student model config (width shrinking)")
    parser.add_argument(
        "--override", action="append", default=[], help="dotted.key=value edit of --config"
    )
    parser.add_argument(
        "--ema", action="store_true", help="with --config: seed from the EMA weights"
    )
    parser.add_argument(
        "--cov",
        help="with --config: residual covariance (training.scripts.residual_cov) to cut "
        "d_model along its principal components",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if (args.layers is None) == (args.config is None):
        parser.error("give exactly one of --layers and --config")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = payload.get("model", payload)
    ema = payload.get("ema")
    if args.config is None:
        shrunk, keep = shrink(model, args.layers)
        if ema:
            ema = shrink(ema, args.layers)[0]
        print(f"kept teacher layers {keep}")
    else:
        if args.ema and ema:
            model = {**model, **ema}
        overrides = {k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in args.override)}
        dst, head_dim = flow_lm_shapes(args.config, overrides)
        src = {k.removeprefix("flow_lm."): v for k, v in model.items() if k.startswith("flow_lm.")}
        src, keep = shrink(src, len(layer_prefixes(dst)))
        cov = torch.load(args.cov, weights_only=True)["cov"] if args.cov else None
        seeded = shrink_width(src, dst, head_dim, cov)
        shrunk = {k: v for k, v in model.items() if not k.startswith("flow_lm.")}
        shrunk.update({f"flow_lm.{k}": v for k, v in seeded.items()})
        ema = None
        print(f"kept teacher layers {keep}; not seeded: {sorted(set(dst) - set(seeded))}")
    out = {"step": 0, "model": shrunk, "optimizer": None, "ema": ema}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
