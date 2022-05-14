import torch
import torchvision
import torchvision.transforms as transforms
import librosa.display
import matplotlib.pyplot as plt
import io
from pytorch_lightning import Callback

import melgan.modules as modules
import hparams
import dataset

class ImageLogger(Callback):
    def __init__(self, sample_x_path, log_interval: int = 100):
        self.log_interval = log_interval
        self.sample_x_path = sample_x_path
        self.vocoder = modules.Generator()
        self.vocoder.load_state_dict(torch.load("melgan/model.pt"))
        self.wav2mel = modules.Wav2Mel()
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        global_step = trainer.global_step
        if global_step % self.log_interval == 0:
            x_raw = dataset.readwavfile(self.sample_x_path)
            x = self.wav2mel(x_raw).unsqueeze(0).float().to(batch[0].device)

            y_hat = pl_module(x).detach().cpu()
            transform = transforms.Resize((80, y_hat.shape[2]))
            out_data = y_hat.numpy()

            # add image to tensorboard
            fig, ax = plt.subplots()
            img = librosa.display.specshow(out_data[0], x_axis='time',y_axis='mel', sr=hparams.sr, fmax=hparams.fmax, fmin=hparams.fmin, ax=ax, hop_length=hparams.hop_size, cmap='magma')
            fig.colorbar(img, ax=ax, format='%f dB')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            
            byte_tensor = torch.as_tensor(list(buf.getvalue()), dtype=torch.uint8)
            img = torchvision.io.decode_image(byte_tensor)
            img = img.unsqueeze(0)

            grid = torchvision.utils.make_grid(img)
            trainer.logger.experiment.add_image('generated-y', grid, global_step=global_step)

            # add audio to tensorboard
            fig, ax = plt.subplots()
            img = librosa.display.specshow(x.cpu().numpy()[0], x_axis='time',y_axis='mel', sr=hparams.sr, fmax=hparams.fmax, fmin=hparams.fmin, ax=ax, hop_length=hparams.hop_size, cmap='magma')
            fig.colorbar(img, ax=ax, format='%f dB')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            
            byte_tensor = torch.as_tensor(list(buf.getvalue()), dtype=torch.uint8)
            img = torchvision.io.decode_image(byte_tensor)
            img = img.unsqueeze(0)

            grid = torchvision.utils.make_grid(img)
            trainer.logger.experiment.add_image('original-x', grid, global_step=global_step)


            # add audio to tensorboard
            mel = transform(y_hat)[0]

            with torch.no_grad():
                audio = self.vocoder(mel)
            audio.permute(1,0)

            trainer.logger.experiment.add_audio('audio', audio, global_step=global_step, sample_rate=hparams.sr)