import torch
import torch.nn as nn
from torch.nn.modules.activation import GLU
import hparams


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()
    def forward(self, x):
        return x * torch.sigmoid(x)

class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1,3), stride=(1,1), padding=(0,1), padding_mode='reflect'),
            nn.InstanceNorm2d(256),
            GLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1,3), stride=(1,1), padding=(0,1), padding_mode='reflect'),
            nn.InstanceNorm2d(256),
        )
    
    def forward(self, x):
        return x + self.resblock(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.gen1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=128, kernel_size=(5,15), stride=(1,1), padding=(2,7), padding_mode='reflect'),
            GLU(),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5,5), stride=(2,2), padding=(2,2), padding_mode='reflect'),
            nn.InstanceNorm2d(256),
            GLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5,5), stride=(2,2), padding=(2,2), padding_mode='reflect'),
            nn.InstanceNorm2d(256),
            GLU(),

            nn.Flatten(1,2),
        )

        self.gen2 = nn.Sequential(
            nn.Conv2d(in_channels=256*hparams.n_mels//4, out_channels=256, kernel_size=(1,1), stride=(1,1), padding=0, padding_mode='reflect'),
            nn.InstanceNorm2d(256),

            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock(),

            nn.Conv2d(in_channels=256, out_channels=256*hparams.n_mels//4, kernel_size=(1,1), stride=(1,1), padding=0, padding_mode='reflect'),
            nn.InstanceNorm2d(256*hparams.n_mels//4),
        )

        self.gen3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(5,5), stride=(1,1), padding=(2,2), padding_mode='reflect'),
            nn.PixelShuffle(2),
            nn.InstanceNorm2d(256),
            GLU(),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(5,5), stride=(1,1), padding=(2,2), padding_mode='reflect'),
            nn.PixelShuffle(2),
            nn.InstanceNorm2d(128),
            GLU(),

            nn.Conv2d(in_channels=128, out_channels=35, kernel_size=(5,15), stride=(1,1), padding=(2,7), padding_mode='reflect'),
            nn.Conv2d(in_channels=35, out_channels=1, kernel_size=1, stride=1, padding=0, padding_mode='reflect'),
        )
    
    # (B, N_MEL, LENGTH) -> (B, N_MEL, LENGTH)
    def forward(self, mel, mask):
        x = mel * mask
        x = mel.unsqueeze(1)
        mask = mask.unsqueeze(1)
        x = torch.concat([x, mask], dim=1)
        x = self.gen1(x)
        x = x.unsqueeze(2)
        x = self.gen2(x)
        x = x.view(x.shape[0], 256, hparams.n_mels//4, -1)
        x = self.gen3(x)
        x = x.squeeze(1)
        return x

        

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3,3), stride=(1,1)),
            GLU(),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=(2,2), padding=1),
            nn.InstanceNorm2d(256),
            GLU(),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=(2,2), padding=1),
            nn.InstanceNorm2d(512),
            GLU(),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3,3), stride=(2,2), padding=1),
            nn.InstanceNorm2d(1024),
            GLU(),

            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3,5), stride=(1,1), padding=(1,2)),
            nn.InstanceNorm2d(1024),
            GLU(),

            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=(1,3), stride=(1,1), padding=(0,1)),
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.disc(x)
        return x