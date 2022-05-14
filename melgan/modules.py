import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import numpy as np
from librosa.filters import mel as librosa_mel_fn

import hparams

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))

class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(dilation),
            WNConv1d(dim, dim, kernel_size=3, dilation=dilation),
            nn.LeakyReLU(0.2),
            WNConv1d(dim, dim, kernel_size=1),
        )
        self.shortcut = WNConv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)

class Generator(nn.Module):
    def __init__(self, input_size = 80, ngf = 32, n_residual_layers = 3):
        super().__init__()
        ratios = [8, 8, 2, 2]
        self.hop_length = np.prod(ratios)
        mult = int(2 ** len(ratios))

        model = [
            nn.ReflectionPad1d(3),
            WNConv1d(input_size, mult * ngf, kernel_size=7, padding=0),
        ]

        # Upsample to raw audio scale
        for i, r in enumerate(ratios):
            model += [
                nn.LeakyReLU(0.2),
                WNConvTranspose1d(
                    mult * ngf,
                    mult * ngf // 2,
                    kernel_size=r * 2,
                    stride=r,
                    padding=r // 2 + r % 2,
                    output_padding=r % 2,
                ),
            ]

            for j in range(n_residual_layers):
                model += [ResnetBlock(mult * ngf // 2, dilation=3 ** j)]

            mult //= 2

        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(ngf, 1, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)
        self.apply(weights_init)

    def forward(self, x):
        return self.model(x).squeeze(1)

class Wav2Mel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,
            audio,
            sampling_rate=hparams.sr,
            fft_size=hparams.fft_size,
            hop_size=hparams.hop_size,
            win_length=hparams.win_size,
            window="hann",
            num_mels=hparams.n_mels,
            fmin=hparams.fmin,
            fmax=hparams.fmax
        ):
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        fft = torch.stft(
            audio,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_length,
            pad_mode="reflect",
            window=torch.hann_window(win_length).float(),
            center=False,
            return_complex=False
        )
        mel_basis = torch.from_numpy(librosa_mel_fn(sampling_rate, fft_size, num_mels, fmin, fmax))
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        mel_output = torch.matmul(mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        return log_mel_spec