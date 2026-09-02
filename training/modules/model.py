"""Trainable wrapper around the pocket-tts FlowLM.

Reuses the pocket-tts modules (StreamingTransformer with model_state=None as a
plain causal transformer, LUTConditioner, SimpleMLPAdaLN-style head, Mimi) so
that checkpoints trained here are directly loadable by pocket-tts inference.

Sequence layout (matches pocket-tts generation exactly):
    [bos_before_voice, voice latents @ speaker_proj, text embeddings, audio...]
with the audio stream teacher-forced through input_linear (BOS latent first).
CFG dropout removes the text and/or voice segments entirely.
"""

from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from pocket_tts.models.flow_lm import FlowLMModel
from pocket_tts.modules.stateful_module import ModelState, increment_steps, init_states

from ..args import TrainArgs
from .conditioner import build_sequences_with_conditions
from .samplers import FlowType
from .utils import set_state_padding, stamp_state_names


class TrainableTTS(nn.Module):
    # Frozen teacher for distillation runs, kept out of the submodule registry
    # (see build_models); None otherwise.
    distill_teacher: FlowLMModel | None

    def __init__(self, flow_lm: FlowLMModel, flow: FlowType, args: TrainArgs):
        super().__init__()
        self.flow_lm = flow_lm
        self.flow = flow
        self.args = args
        self.__dict__["distill_teacher"] = None  # set by build_models; not a submodule
        stamp_state_names(self.flow_lm)

    @property
    def ldim(self) -> int:
        return self.flow_lm.ldim

    def _update_latent_stats(self, latents: torch.Tensor, mask: torch.Tensor):
        fl = self.flow_lm
        with torch.no_grad():
            sel = latents[mask]
            gamma = self.args.stats_ema_decay
            fl.emb_mean.mul_(gamma).add_(sel.mean(dim=0), alpha=1 - gamma)
            fl.emb_std.mul_(gamma).add_(sel.std(dim=0), alpha=1 - gamma)

    def forward(
        self,
        latents: torch.Tensor,  # [B, T, C] raw codec latents
        mask: torch.Tensor,  # [B, T] valid positions
        text_tokens: list[torch.Tensor],
        voice_latents: torch.Tensor,  # [B, Tv, C] raw codec latents
        update_stats: bool = False,
        num_voice_prompt_frames: torch.Tensor
        | None = None,  # [B] valid voice frames (rest is padding)
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        fl = self.flow_lm
        _, T, C = latents.shape
        if update_stats and self.training:
            self._update_latent_stats(latents, mask)
        normalized_latents = (latents - fl.emb_mean) / fl.emb_std

        def backbone_z(
            module: FlowLMModel, cfg_dropout: bool, force_null: bool = False
        ) -> torch.Tensor:
            x, prefix_lengths = build_sequences_with_conditions(
                self.args,
                normalized_latents,
                text_tokens,
                voice_latents,
                cfg_dropout=cfg_dropout,
                num_voice_prompt_frames=num_voice_prompt_frames,
                fl=module,
                force_null=force_null,
            )
            out = module.out_norm(module.transformer(x, model_state=None))
            idx = prefix_lengths[:, None] + torch.arange(T, device=out.device)[None, :]
            return out.gather(1, idx[:, :, None].expand(-1, -1, out.shape[-1]))

        z = backbone_z(fl, cfg_dropout=self.training)  # [B, T, dim]

        if self.distill_teacher is not None:
            # Latent CFG distillation: the student's backbone regresses onto the
            # teacher's guidance-combined output; flow/EOS heads stay frozen.
            teacher = self.distill_teacher.to(latents.device)
            with torch.no_grad():
                z_cond = backbone_z(teacher, cfg_dropout=False)
                z_null = backbone_z(teacher, cfg_dropout=False, force_null=True)
                z_t = z_null + self.args.distill_cfg_coef * (z_cond - z_null)
            shifted_mask = torch.cat([mask[:, :1], mask[:, :-1]], dim=1)
            denom = shifted_mask.sum().clamp(min=1)
            loss = ((z - z_t).square().mean(dim=-1) * shifted_mask).sum() / denom
            return loss, {"distill_mse": loss.detach(), "loss": loss.detach()}

        # EOS loss: 1 on the first invalid position, computed over valid
        # positions plus that first invalid one (mask shifted right).
        is_eos = ~mask
        is_eos[:, 0] = False
        eos_logits = fl.out_eos(z).squeeze(-1)
        shifted_mask = torch.cat([mask[:, :1], mask[:, :-1]], dim=1)
        eos_loss = is_eos * F.softplus(-eos_logits) + ~is_eos * F.softplus(eos_logits)
        eos_loss = (eos_loss * shifted_mask).sum() / shifted_mask.sum().clamp(min=1)

        # Flow loss on valid positions, optionally with several noise draws per
        # position (flow_batch_multiplier).
        fbm = self.args.flow_batch_multiplier
        z_flat = z.unsqueeze(0).expand(fbm, -1, -1, -1).reshape(-1, z.shape[-1])
        target_flat = normalized_latents.unsqueeze(0).expand(fbm, -1, -1, -1).reshape(-1, C)
        mask_flat = mask.unsqueeze(0).expand(fbm, -1, -1).reshape(-1)
        sel_z = z_flat[mask_flat]
        sel_target = target_flat[mask_flat]
        noise = torch.randn_like(sel_target)
        flow_loss, metrics, _ = self.flow.loss(partial(fl.flow_net, sel_z), noise, sel_target)
        flow_loss = flow_loss.mean()

        loss = flow_loss + self.args.eos_loss_weight * eos_loss
        # Detach every metric: they are logging-only, and non-detached extra
        # outputs give the compiled backward immutable ZeroTensor grads.
        metrics = {k: v.detach() if torch.is_tensor(v) else v for k, v in metrics.items()}
        metrics.update(flow_loss=flow_loss.detach(), eos_loss=eos_loss.detach(), loss=loss.detach())
        return loss, metrics

    @torch.no_grad()
    def generate(
        self,
        text_tokens: list[torch.Tensor],  # per-row token tensors, lengths may differ
        voice_latents: list[torch.Tensor],  # per-row [Tv, C], lengths may differ
        max_frames: int = 375,
        temp: float = 0.8,
        n_steps: int = 1,
        cfg_coef: float = 1.0,
        eos_threshold: float = -1.0,
        eos_countdown: int = 1,
    ) -> list[torch.Tensor]:
        """Sample a batch of utterances with arbitrary prefix lengths.

        Rows are right-aligned (padded on the left) so every row's first audio
        frame lands on the same step; the padding is masked out via the
        per-row `pad` entry in the streaming state.
        """
        fl = self.flow_lm
        device = fl.bos_emb.device
        was_training = self.training
        self.eval()
        B = len(text_tokens)

        rows = []
        for tokens, voice in zip(text_tokens, voice_latents, strict=True):
            t_emb = fl.conditioner(tokens[None].to(device))[0]
            v_emb = F.linear(
                voice[None].to(device, fl.speaker_proj_weight.dtype), fl.speaker_proj_weight
            )[0]
            rows.append(torch.cat([fl.bos_before_voice[0], v_emb, t_emb], dim=0))
        widths = [r.shape[0] for r in rows]
        width = max(widths)
        cond = torch.zeros(B, width, rows[0].shape[-1], device=device, dtype=rows[0].dtype)
        for b, r in enumerate(rows):
            cond[b, width - r.shape[0] :] = r  # right-align
        pad_cond = torch.tensor([width - w for w in widths], device=device, dtype=torch.long)

        prefixes = [cond]
        pads = [pad_cond]
        if cfg_coef != 1.0:
            prefixes.append(fl.bos_before_voice.expand(B, -1, -1))
            pads.append(torch.zeros(B, device=device, dtype=torch.long))

        states: list[ModelState] = []
        for p, pd in zip(prefixes, pads, strict=True):
            st = init_states(fl, B, p.shape[1] + max_frames + 2)
            set_state_padding(st, pd)
            states.append(st)
        first_step = True  # the prefixes are fed along with the first latent

        def run(x_lat: torch.Tensor) -> torch.Tensor:
            nonlocal first_step
            zs = []
            for i, st in enumerate(states):
                inp = fl.input_linear(x_lat)
                if first_step:
                    inp = torch.cat([prefixes[i], inp], dim=1)
                out = fl.out_norm(fl.transformer(inp, st))
                increment_steps(fl, st, increment=inp.shape[1])
                zs.append(out[:, -1].float())
            first_step = False
            return zs[0] if len(zs) == 1 else zs[1] + cfg_coef * (zs[0] - zs[1])

        x_lat = fl.bos_emb.view(1, 1, -1).expand(B, 1, -1)
        frames: list[torch.Tensor] = []
        countdown = torch.full((B,), -1, device=device, dtype=torch.long)
        ends = torch.full((B,), max_frames, device=device, dtype=torch.long)
        for t in range(max_frames):
            z = run(x_lat)
            noise = temp**0.5 * torch.randn(B, self.ldim, device=device, dtype=z.dtype)
            latent = self.flow.decode(partial(fl.flow_net, z), noise, n_steps)
            fired = (fl.out_eos(z).squeeze(-1) > eos_threshold) & (countdown < 0)
            countdown = torch.where(fired, torch.full_like(countdown, eos_countdown), countdown)
            countdown = torch.where(countdown > 0, countdown - 1, countdown)
            done = countdown == 0
            # A finished row keeps its frames up to t: the frame on which the
            # countdown expires is not appended.
            ends = torch.where(done, torch.minimum(ends, torch.full_like(ends, t)), ends)
            if bool(done.all()):
                break
            frames.append(latent)
            x_lat = latent[:, None, :].to(fl.bos_emb.dtype)
        self.train(was_training)

        if not frames:  # every row emitted EOS on the first step
            return [torch.zeros(0, self.ldim, device=device) for _ in range(B)]
        stacked = torch.stack(frames, dim=1)
        return [stacked[b, : int(ends[b])] * fl.emb_std + fl.emb_mean for b in range(B)]
