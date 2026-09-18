"""Neural codec latent spaces used as the attack space (EnCodec, DAC).

Both wrappers map 16 kHz waveforms to a continuous latent ``[B, C, F]`` and
back, so the attacker never sees the codec's native sample rate.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio.functional as AF

from hackpatch.utils.audio import SR


class LatentCodec(nn.Module):
    """Frozen codec with ``encode``/``decode`` over 16 kHz audio.

    Built from the ``codec`` config section.

    Args:
        name: ``encodec_24khz`` or ``dac_16khz``.
        quantize: Decode from the quantized latent (True) or the continuous
            encoder output (False). The attack perturbs whichever is used.
        encodec_bandwidth: kbps for EnCodec quantization when ``quantize``.
    """

    def __init__(self, name: str = "encodec_24khz", quantize: bool = False, encodec_bandwidth: float = 24.0):
        super().__init__()
        self.name, self.quantize, self.bandwidth = name, quantize, encodec_bandwidth
        if name == "encodec_24khz":
            from transformers import EncodecModel

            self.model = EncodecModel.from_pretrained("facebook/encodec_24khz")
            self.codec_sr = 24000
        elif name == "dac_16khz":
            import dac

            self.model = dac.DAC.load(dac.utils.download(model_type="16khz"))
            self.codec_sr = 16000
        else:
            raise ValueError(f"unknown codec {name}")
        self.model.eval().requires_grad_(False)

    @property
    def hop(self) -> int:
        """Input samples (at 16 kHz) per latent frame."""
        return 320 * SR // self.codec_sr if self.name == "encodec_24khz" else 320

    @torch.no_grad()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """Encodes ``[B, T]`` 16 kHz audio to a latent ``[B, C, F]``."""
        x = wav.unsqueeze(1)
        if self.codec_sr != SR:
            x = AF.resample(x, SR, self.codec_sr)
        if self.name == "encodec_24khz":
            z = self.model.encoder(x)
            if self.quantize:
                codes = self.model.quantizer.encode(z, bandwidth=self.bandwidth)
                z = self.model.quantizer.decode(codes)
            return z
        x = self.model.preprocess(x, self.codec_sr)
        z = self.model.encoder(x)
        if self.quantize:
            z = self.model.quantizer(z)[0]
        return z

    @torch.no_grad()
    def decode(self, z: torch.Tensor, length: int) -> torch.Tensor:
        """Decodes a latent to ``[B, length]`` 16 kHz audio."""
        y = self.model.decoder(z)
        if self.codec_sr != SR:
            y = AF.resample(y, self.codec_sr, SR)
        y = y[..., :length]
        if y.shape[-1] < length:
            y = torch.nn.functional.pad(y, (0, length - y.shape[-1]))
        return y.squeeze(1).clamp(-1, 1)
