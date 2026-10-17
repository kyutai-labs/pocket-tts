"""Reward fine-tuning (ReFL / DRaFT style): differentiable rewards through the sampler head and
the Mimi decoder.

The model is rolled out without gradients. The generated frames are then the teacher-forced
context of one backbone pass with gradients, the sampler head is re-applied on the rollout's own
noise (its output is the rollout itself, now differentiable), the frozen Mimi decoder turns it
into audio, and the rewards are backpropagated: UTMOS, and the CTC log-likelihood of the text
under a frozen character-level ASR.

Only rows whose text the ASR can score are rolled out: it cannot score digits or symbols, whose
spoken form is unknown.
"""

import random
import re
from collections.abc import Callable
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

from pocket_tts.models.mimi import MimiModel, build_mimi
from pocket_tts.modules.stateful_module import init_states
from pocket_tts.utils.config import load_config
from pocket_tts.utils.utils import download_if_necessary
from pocket_tts.utils.weights_loading import get_mimi_state_dict
from training.args import ReflArgs
from training.eval.librispeech import MIN_FRAMES
from training.modules.model import TrainableTTS
from training.modules.utils import disable_grad, stamp_state_names

UNSCORABLE = re.compile(r"[\d@/_#%$&+=*<>\\|~^]")


class Rewards:
    """The frozen scorers, and the Mimi that decodes the rollouts for them."""

    def __init__(self, ra: ReflArgs, device: torch.device, mimi: MimiModel):
        try:
            import torchaudio
            from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC
            from utmos_pytorch import UTMOSScoreTorch
        except ImportError as e:
            raise ImportError(
                f"refl needs utmos-pytorch, transformers and torchaudio ({e}); they are in the "
                "dev dependency group: uv sync"
            ) from e

        self.mimi = mimi
        if ra.reward_decoder:
            config = load_config(Path(ra.reward_decoder))
            weights = config.mimi.weights_path or config.weights_path
            assert weights, f"{ra.reward_decoder} names no Mimi weights"
            self.mimi = build_mimi(config.mimi)
            self.mimi.load_state_dict(get_mimi_state_dict(download_if_necessary(weights)))
            self.mimi.to(device).eval()
            stamp_state_names(self.mimi)
            disable_grad(self.mimi)
        self.resample = partial(
            torchaudio.functional.resample, orig_freq=self.mimi.sample_rate, new_freq=16000
        )
        self.utmos = UTMOSScoreTorch(device=str(device)).model
        self.ctc = Wav2Vec2ForCTC.from_pretrained(ra.ctc_model).to(device).eval()  # ty: ignore[invalid-argument-type]
        disable_grad(self.utmos)
        disable_grad(self.ctc)
        # cuDNN's RNN backward refuses eval mode; the single-layer LSTM has no dropout.
        for m in self.utmos.modules():
            if isinstance(m, torch.nn.LSTM):
                m.train()
        self.vocab = Wav2Vec2CTCTokenizer.from_pretrained(ra.ctc_model).get_vocab()
        self.blank = self.ctc.config.pad_token_id

    def utmos_score(self, wav16: torch.Tensor) -> torch.Tensor:
        x = wav16[None, None]
        one = torch.ones(1, dtype=torch.int, device=x.device)
        return self.utmos(x, domains=0 * one, judge_id=288 * one).mean() * 2 + 3

    def ctc_targets(self, text: str) -> list[int] | None:
        """Character targets; None when the text has digits or symbols."""
        if UNSCORABLE.search(text):
            return None
        t = re.sub(r"[^A-Z']+", " ", text.upper().replace("’", "'")).strip()
        return [self.vocab["|" if c == " " else c] for c in t] or None

    def ctc_loss(self, wav16: torch.Tensor, tgt: list[int]) -> torch.Tensor:
        """Per-character CTC negative log-likelihood of the target text."""
        x = (wav16 - wav16.mean()) / torch.sqrt(wav16.var() + 1e-7)
        logp = self.ctc(x[None]).logits.float().log_softmax(-1)[0]
        return F.ctc_loss(
            logp[:, None],
            torch.tensor(tgt, device=logp.device)[None],
            torch.tensor([logp.shape[0]]),
            torch.tensor([len(tgt)]),
            blank=self.blank,
            reduction="sum",
            zero_infinity=True,
        ) / len(tgt)


def refl_loss(
    model: TrainableTTS,
    decode_text: Callable[[list[int]], str],
    text_tokens: list[torch.Tensor],
    voice: torch.Tensor,  # [B, Tv, C] voice prompt latents
    num_voice_prompt_frames: torch.Tensor,  # [B]
    voice_ids: torch.Tensor | None,  # [B] built-in voices, with a voice table
    mask: torch.Tensor,  # [B, T] the real rows' valid frames
    rewards: Rewards,
    ra: ReflArgs,
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
    """Roll out up to ra.rows scorable rows; return (weighted reward loss or None, stats)."""
    fl, mimi = model.flow_lm, rewards.mimi
    device = voice.device
    targets = [rewards.ctc_targets(decode_text(t.tolist())) for t in text_tokens]
    scorable = [(b, t) for b, t in enumerate(targets) if t is not None][: ra.rows]
    rows = [b for b, _ in scorable]
    stats = {"refl_rows": torch.tensor(float(len(rows)))}
    if not rows:
        return None, stats
    R = len(rows)
    toks = [text_tokens[b] for b in rows]
    voice, nvf = voice[rows], num_voice_prompt_frames[rows]
    vids = voice_ids[rows].to(device) if voice_ids is not None else None
    max_frames = int(ra.max_sec * mimi.frame_rate)
    noises: list[torch.Tensor] = []
    outs = model.generate(
        toks,
        [voice[b, : int(nvf[b])] for b in range(R)],
        max_frames=max_frames,
        temp=ra.temp,
        voice_ids=vids.tolist() if vids is not None else None,
        noises=noises,
    )
    lens = [o.shape[0] for o in outs]
    ok = [MIN_FRAMES <= n < max_frames for n in lens]
    gen = torch.tensor(lens, device=device, dtype=torch.float)
    stats["refl_ok"] = torch.tensor(sum(ok) / R)  # rows that emitted EOS in time
    stats["refl_len_ratio"] = (gen / mask[rows].sum(1)).mean()  # rollout / real utterance length
    if not any(ok):
        return None, stats

    T = max(lens)
    lat = torch.stack([F.pad(o, (0, 0, 0, T - o.shape[0])) for o in outs])
    norm = (lat - fl.emb_mean) / fl.emb_std
    noise = torch.stack(noises[:T], dim=1)
    lens_t = torch.tensor(lens, device=device)
    valid = torch.arange(T, device=device)[None, :] < lens_t[:, None]
    z = model.backbone_z(fl, norm, toks, voice, nvf, vids, cfg_dropout=False)
    x = model.flow.decode(partial(fl.flow_net, z.float()), noise.float(), 1)
    stats["refl_parity"] = (x.detach() - norm)[valid].abs().max()  # ~0: x is the rollout
    # Frames past a row's end repeat its last frame; the decoder is causal, so they
    # cannot reach the row's own samples, which are cut at its length.
    last = norm[torch.arange(R, device=device), (lens_t - 1).clamp(min=0)]
    x = torch.where(valid[..., None], x, last[:, None])
    ratio = round(mimi.encoder_frame_rate / mimi.frame_rate)
    state = init_states(mimi, R, (T + MIN_FRAMES) * ratio)
    audio = mimi.decode_from_latent(x * fl.emb_std + fl.emb_mean, state)[:, 0]

    moses, ctcs = [], []
    for b in range(R):
        if not ok[b]:
            continue
        w16 = rewards.resample(
            audio[b, random.randint(0, ra.max_shift) : lens[b] * mimi.frame_size]
        )
        with torch.set_grad_enabled(ra.utmos_weight > 0):
            moses.append(rewards.utmos_score(w16))
        with torch.set_grad_enabled(ra.ctc_weight > 0):
            ctcs.append(rewards.ctc_loss(w16, scorable[b][1]))
    mos, ctc = torch.stack(moses).mean(), torch.stack(ctcs).mean()
    stats["refl_utmos"], stats["refl_ctc"] = mos.detach(), ctc.detach()
    return ra.ctc_weight * ctc - ra.utmos_weight * mos, stats
