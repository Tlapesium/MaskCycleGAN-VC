import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import random
from itertools import chain

import maskcyclegan.modules
import hparams


class MaskCycleGAN_VC(LightningModule):
    def __init__(self):
        super().__init__()
        self.generator_XY = maskcyclegan.modules.Generator()
        self.generator_YX = maskcyclegan.modules.Generator()
        self.discriminator_X = maskcyclegan.modules.Discriminator()
        self.discriminator_Y = maskcyclegan.modules.Discriminator()
        self.discriminator_X2 = maskcyclegan.modules.Discriminator()
        self.discriminator_Y2 = maskcyclegan.modules.Discriminator()
    
        self.loss_fnc_MSE = torch.nn.MSELoss()
        self.loss_fnc_L1 = torch.nn.L1Loss()

    def _get_mask(self, x):
        mask = torch.ones_like(x)
        a = random.randint(0, hparams.learn_length//2) # 割合
        b = random.randint(0, hparams.learn_length-a) # 開始
        mask[:, b:b+a] = 0
        return mask

    def forward(self, z):
        return self.generator_XY(z, torch.ones_like(z))

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch

        x_l = random.randrange(0, x.shape[2] - hparams.learn_length)
        y_l = random.randrange(0, y.shape[2] - hparams.learn_length)

        x = x[:, :, x_l:x_l+hparams.learn_length]
        y = y[:, :, y_l:y_l+hparams.learn_length]

        # Train Generator
        if optimizer_idx == 0:
            self.generator_XY.train()
            self.generator_YX.train()
            self.discriminator_X.eval()
            self.discriminator_Y.eval()
            self.discriminator_X2.eval()
            self.discriminator_Y2.eval()

            fake_x = self.generator_YX(y, self._get_mask(y))
            fake_y = self.generator_XY(x, self._get_mask(x))

            fake_x2 = self.generator_YX(fake_y, torch.ones_like(fake_y))
            fake_y2 = self.generator_XY(fake_x, torch.ones_like(fake_x))

            fake_x_y = self.generator_XY(y, torch.ones_like(y))
            fake_y_x = self.generator_YX(x, torch.ones_like(x))

            pred_fake_x = self.discriminator_X(fake_x)
            pred_fake_y = self.discriminator_Y(fake_y)

            pred_fake_x2 = self.discriminator_X2(fake_x2)
            pred_fake_y2 = self.discriminator_Y2(fake_y2)

            loss_adv = self.loss_fnc_MSE(pred_fake_x, torch.ones_like(pred_fake_x)) + self.loss_fnc_MSE(pred_fake_y, torch.ones_like(pred_fake_y))
            loss_adv += self.loss_fnc_MSE(pred_fake_x2, torch.ones_like(pred_fake_x2)) + self.loss_fnc_MSE(pred_fake_y2, torch.ones_like(pred_fake_y2))
            loss_cycle = self.loss_fnc_L1(fake_x2, x) + self.loss_fnc_L1(fake_y2, y)
            loss_identity = self.loss_fnc_L1(fake_x_y, y) + self.loss_fnc_L1(fake_y_x, x)

            loss_G = loss_adv + loss_cycle * hparams.l_cycle + loss_identity * (hparams.l_ident if self.global_step < hparams.ident_end else 0)

            self.log("loss_G", loss_G, prog_bar=True)
            self.log("loss_adv", loss_adv)
            self.log("loss_cycle", loss_cycle)
            self.log("loss_identity", loss_identity)

            output = {'loss': loss_G}
            return output
        
        if optimizer_idx >= 1:
            discriminators = [None, self.discriminator_X, self.discriminator_Y, self.discriminator_X2, self.discriminator_Y2]
            self.generator_XY.eval()
            self.generator_YX.eval()
            for i in range(1, len(discriminators)):
                if i == optimizer_idx: discriminators[i].train()
                else: discriminators[i].eval()
                    
            fake_x = self.generator_YX(y, self._get_mask(y))
            fake_y = self.generator_XY(x, self._get_mask(x))

            fake_x2 = self.generator_YX(fake_y, torch.ones_like(fake_y))
            fake_y2 = self.generator_XY(fake_x, torch.ones_like(fake_x))

            reals = [None, x, y, x, y]
            fakes = [None, fake_x, fake_y, fake_x2, fake_y2]

            pred_real = discriminators[optimizer_idx](reals[optimizer_idx])
            pred_fake = discriminators[optimizer_idx](fakes[optimizer_idx])

            loss_D = self.loss_fnc_MSE(pred_real, torch.ones_like(pred_real)) + self.loss_fnc_MSE(pred_fake, torch.zeros_like(pred_fake))

            self.log("loss_D", loss_D, prog_bar=True)

            output = {'loss': loss_D}
            return output

    def configure_optimizers(self):
        gen_lr = hparams.gen_lr
        disc_lr = hparams.disc_lr
        beta = hparams.beta

        optim_G = hparams.g_optimizer(chain(self.generator_XY.parameters(), self.generator_YX.parameters()), lr=gen_lr)
        optim_D_X = hparams.d_optimizer(self.discriminator_X.parameters(), lr=disc_lr, betas=beta)
        optim_D_Y = hparams.d_optimizer(self.discriminator_Y.parameters(), lr=disc_lr, betas=beta)
        optim_D_X2 = hparams.d_optimizer(self.discriminator_X2.parameters(), lr=disc_lr, betas=beta)
        optim_D_Y2 = hparams.d_optimizer(self.discriminator_Y2.parameters(), lr=disc_lr, betas=beta)

        return [optim_G, optim_D_X, optim_D_Y, optim_D_X2, optim_D_Y2], []