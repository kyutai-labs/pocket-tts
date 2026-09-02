import logging
from typing import NoReturn

import torch
from torch import nn

from pocket_tts.modules.conv import pad_for_conv1d
from pocket_tts.modules.dummy_quantizer import DummyQuantizer
from pocket_tts.modules.resample import ConvDownsample1d, ConvTrUpsample1d
from pocket_tts.modules.seanet import SEANetDecoder, SEANetEncoder
from pocket_tts.modules.stateful_module import ModelState
from pocket_tts.modules.transformer import ProjectedTransformer
from pocket_tts.utils.config import MimiConfig

logger = logging.getLogger()


class MimiModel(nn.Module):
    def __init__(
        self,
        encoder: SEANetEncoder,
        decoder: SEANetDecoder,
        quantizer: DummyQuantizer,
        frame_rate: float,
        encoder_frame_rate: float,
        sample_rate: int,
        channels: int,
        inner_dim: int | None,
        outer_dim: int | None,
        encoder_transformer: ProjectedTransformer,
        decoder_transformer: ProjectedTransformer,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_transformer = encoder_transformer
        self.decoder_transformer = decoder_transformer
        self.quantizer = quantizer
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.channels = channels
        self.encoder_frame_rate = encoder_frame_rate

        # We will need the dimension for the resampling. In general the encoder will be a SeanetEncoder
        # which exposes a `dimension` attribute.
        dimension = encoder.dimension
        assert isinstance(dimension, int), (
            f"Dimension should be int, got {dimension} of type {type(dimension)}."
        )
        self.dimension = dimension

        if encoder_frame_rate != frame_rate:
            assert self.encoder_frame_rate > self.frame_rate, "Cannot upsample with conv."
            downsample_stride = self.encoder_frame_rate / self.frame_rate
            assert downsample_stride == int(downsample_stride), (
                f"Only integer strides are supported, got {downsample_stride}"
            )

            # v2 of models
            self.downsample = ConvDownsample1d(
                int(downsample_stride), dimension=dimension, out_dimension=inner_dim
            )
            self.upsample = ConvTrUpsample1d(
                int(downsample_stride), dimension=dimension, in_dimension=outer_dim
            )

    @property
    def frame_size(self) -> int:
        return int(self.sample_rate / self.frame_rate)

    def _to_framerate(self, x: torch.Tensor) -> torch.Tensor:
        # Convert from the encoder frame rate to the overall framerate.
        _, _, length = x.shape
        frame_rate = self.encoder_frame_rate
        new_frame_rate = self.frame_rate
        if frame_rate == new_frame_rate:
            return x
        return self.downsample(x, model_state=None)

    def _to_encoder_framerate(self, x: torch.Tensor, mimi_state: ModelState) -> torch.Tensor:
        # Convert from overall framerate to the encoder frame rate.
        _, _, length = x.shape
        frame_rate = self.encoder_frame_rate
        new_frame_rate = self.frame_rate
        if frame_rate == new_frame_rate:
            return x
        return self.upsample(x, mimi_state)

    def forward(self, x: torch.Tensor) -> NoReturn:
        raise NotImplementedError()

    def decode_from_latent(self, latent: torch.Tensor, mimi_state: ModelState) -> torch.Tensor:
        """Decodes [B, T, C] quantizer-space latents (as produced by
        encode_to_latent or the flow LM) back to audio."""
        latent = self.quantizer(latent.transpose(-1, -2))
        emb = self._to_encoder_framerate(latent, mimi_state)
        (emb,) = self.decoder_transformer(emb, mimi_state)
        out = self.decoder(emb, mimi_state)
        # out contains extra padding added by the encoder and decoder
        return out

    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Projects a batch of waveforms to unquantized latent space.

        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T].

        Returns:
            Unquantized embeddings of shape [B, T, C].
        """
        assert x.dim() == 3, (
            f"CompressionModel._encode_to_unquantized_latent expects audio of shape [B, C, T] but got {x.shape}"
        )

        frame_size = self.frame_size

        # The underlying convolutions no longer accept partial inputs,
        # `x` needs to be exactly a multiple of the frame size,
        # reproducing the previous padding behavior here.
        x = pad_for_conv1d(x, frame_size, frame_size)
        emb = self.encoder(x, model_state=None)

        (emb,) = self.encoder_transformer(emb, model_state=None)
        emb = self._to_framerate(emb)
        return emb.transpose(-1, -2)


def build_mimi(config: MimiConfig) -> MimiModel:
    """MimiModel from a pocket-tts config's `mimi` section."""
    from pocket_tts.modules import transformer
    from pocket_tts.modules.dummy_quantizer import DummyQuantizer
    from pocket_tts.modules.seanet import SEANetDecoder, SEANetEncoder

    mimi_config = config.model_dump()
    encoder = SEANetEncoder(**mimi_config["seanet"])
    decoder = SEANetDecoder(**mimi_config["seanet"])
    return MimiModel(
        encoder,
        decoder,
        DummyQuantizer(**mimi_config["quantizer"]),
        channels=mimi_config["channels"],
        sample_rate=mimi_config["sample_rate"],
        frame_rate=mimi_config["frame_rate"],
        encoder_frame_rate=mimi_config["sample_rate"] / encoder.hop_length,
        inner_dim=mimi_config["inner_dim"],
        outer_dim=mimi_config["outer_dim"],
        encoder_transformer=transformer.ProjectedTransformer(**mimi_config["transformer"]),
        decoder_transformer=transformer.ProjectedTransformer(**mimi_config["transformer"]),
    )
