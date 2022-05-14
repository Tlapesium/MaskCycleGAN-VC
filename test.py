import argparse
import pathlib
import soundfile
import torchvision.transforms as transforms
import torch

import melgan.modules
import modules
import dataset
import hparams

import onnx
import onnxruntime

def main(args):
    if args.model_path.endswith(".ckpt"):
        wav2mel = melgan.modules.Wav2Mel()
        vocoder = melgan.modules.Generator()
        vocoder.load_state_dict(torch.load("melgan/model.pt"))
        model = modules.MaskCycleGAN_VC().load_from_checkpoint(args.model_path)

        in_file = pathlib.Path(args.wav_path)
        in_data = dataset.readwavfile(in_file)
        in_data = wav2mel(in_data)
        with torch.no_grad():
            if args.x2y:
                result = model.generator_XY(in_data.unsqueeze(0), torch.ones_like(in_data.unsqueeze(0)))
            if args.y2x:
                result = model.generator_YX(in_data.unsqueeze(0), torch.ones_like(in_data.unsqueeze(0)))

        transform = transforms.Resize((80, result.shape[2]))
        mel = transform(result)[0]

        with torch.no_grad():
            audio = vocoder(mel)
        soundfile.write(args.output_path, audio.cpu().numpy()[0], hparams.sr, 'PCM_24')

    elif args.model_path.endswith(".onnx"):
        wav2mel = melgan.modules.Wav2Mel()
        in_file = pathlib.Path(args.wav_path)
        in_data = dataset.readwavfile(in_file)
        in_data = wav2mel(in_data)
        
        onnx_model = onnx.load(args.model_path)
        onnx.checker.check_model(onnx_model)

        session = onnxruntime.InferenceSession(args.model_path)

        ort_inputs = {session.get_inputs()[0].name: in_data.unsqueeze(0).numpy()}
        ort_outs = session.run(None, ort_inputs)
        soundfile.write(args.output_path, ort_outs[0][0], hparams.sr, 'PCM_24')

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--wav_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--x2y", action="store_true")
    parser.add_argument("--y2x", action="store_true")
    args = parser.parse_args()

    assert args.x2y or args.y2x
    main(args)