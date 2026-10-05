import torch
from torch import nn

from pocket_tts.modules.stateful_module import ModelState, StatefulModule


class VoiceLUT(StatefulModule):
    """A voice name lookup table summed into every audio frame of the flow LM.

    Models trained on a closed set of voices (training's `voices`) know each voice by name as
    well as from its prompt. Laid out as audiocraft's LUT conditioner: `embed` has a row per voice
    plus an unused padding row, `output_proj` maps it to d_model, and a voice left unset is
    `learnt_padding` instead (also the CFG null). The projection and padding start at zero, so a
    warm-started model is unchanged at step 0.

    The chosen voice's term lives in the streaming state, so a voice state carries it.
    """

    def __init__(self, names: list[str], dim: int, d_model: int):
        super().__init__()
        self.names = sorted(names)
        self.embed = nn.Embedding(len(self.names) + 1, dim)
        self.output_proj = nn.Linear(dim, d_model, bias=False)
        self.learnt_padding = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.zeros_(self.output_proj.weight)

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"unknown voice {name!r}, expected one of {self.names}") from None

    def forward(self, voice_ids: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        """[B, 1, d_model]: the voice's embedding where `keep`, else the learnt padding."""
        emb = self.output_proj(self.embed(voice_ids))[:, None]
        return torch.where(keep[:, None, None], emb, self.learnt_padding.to(emb.dtype))

    def init_state(self, batch_size: int, sequence_length: int) -> dict[str, torch.Tensor]:
        return {"term": self.learnt_padding.detach().expand(batch_size, 1, -1).clone()}

    def get_state_name(self) -> str:
        if self._module_absolute_name is None:
            raise RuntimeError("VoiceLUT has no absolute name: call stamp_state_names() first")
        return self._module_absolute_name

    def term(self, model_state: ModelState) -> torch.Tensor:
        """The [B, 1, d_model] term to sum: the state's voice, or the learnt padding for a
        state without one (e.g. a voice state exported before the model had a LUT)."""
        name = self._module_absolute_name
        if name is None or name not in model_state:
            return self.learnt_padding
        return model_state[name]["term"]

    def select(self, state: dict[str, torch.Tensor], name: str | None):
        """Set the state's voice, or with None the learnt padding (no voice by name)."""
        if name is None:
            term = self.learnt_padding
        else:
            device = self.embed.weight.device
            term = self.output_proj(self.embed(torch.tensor([self.index(name)], device=device)))
        state["term"] = term.detach().reshape(1, 1, -1).expand_as(state["term"]).clone()
