from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse

import dataset
import callbacks
import modules
import hparams

def main(args):
    ds = dataset.MelDataModule(args.dataset_x, args.dataset_y, hparams.batch_size)
    model = modules.MaskCycleGAN_VC()
    trainer = Trainer(max_epochs=hparams.epochs, gpus=1, logger=True, callbacks=[callbacks.ImageLogger(args.sample_x), ModelCheckpoint("./models/", "mask_cyclegan_vc")])
    trainer.fit(model, ds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_x", type=str, required=True)
    parser.add_argument("--dataset_y", type=str, required=True)
    parser.add_argument("--sample_x", type=str, required=True)
    args = parser.parse_args()
    main(args)