import torch
import torch.nn.functional as F
import numpy as np
from torch import nn


class ConvEncoderBENDR(_BENDREncoder):
    def __init__(self, in_features, encoder_h=256, enc_width=(3, 2, 2, 2, 2, 2),
                 dropout=0., projection_head=False, enc_downsample=(3, 2, 2, 2, 2, 2)):
        super().__init__(in_features, encoder_h)
        self.encoder_h = encoder_h
        if not isinstance(enc_width, (list, tuple)):
            enc_width = [enc_width]
        if not isinstance(enc_downsample, (list, tuple)):
            enc_downsample = [enc_downsample]
        assert len(enc_downsample) == len(enc_width)

        # Centerable convolutions make life simpler
        enc_width = [e if e % 2 else e+1 for e in enc_width]
        self._downsampling = enc_downsample
        self._width = enc_width

        self.encoder = nn.Sequential()
        for i, (width, downsample) in enumerate(zip(enc_width, enc_downsample)):
            self.encoder.add_module("Encoder_{}".format(i), nn.Sequential(
                nn.Conv1d(in_features, encoder_h, width, stride=downsample, padding=width // 2),
                nn.Dropout2d(dropout),
                nn.GroupNorm(encoder_h // 2, encoder_h),
                nn.GELU(),
            ))
            in_features = encoder_h

        if projection_head:
            self.encoder.add_module("projection-1", nn.Sequential(
                nn.Conv1d(in_features, in_features, 1),
                nn.Dropout2d(dropout*2),
                nn.GroupNorm(in_features // 2, in_features),
                nn.GELU()
            ))

    def description(self, sfreq=None, sequence_len=None):
        widths = list(reversed(self._width))[1:]
        strides = list(reversed(self._downsampling))[1:]

        rf = self._width[-1]
        for w, s in zip(widths, strides):
            rf = rf if w == 1 else (rf - 1) * s + 2 * (w // 2)

        desc = "Receptive field: {} samples".format(rf)
        if sfreq is not None:
            desc += ", {:.2f} seconds".format(rf / sfreq)

        ds_factor = np.prod(self._downsampling)
        desc += " | Downsampled by {}".format(ds_factor)
        if sfreq is not None:
            desc += ", new sfreq: {:.2f} Hz".format(sfreq / ds_factor)
        desc += " | Overlap of {} samples".format(rf - ds_factor)
        if sequence_len is not None:
            desc += " | {} encoded samples/trial".format(sequence_len // ds_factor)
        return desc

    def downsampling_factor(self, samples):
        for factor in self._downsampling:
            samples = ceil(samples / factor)
        return samples

    def forward(self, x):
        return self.encoder(x)