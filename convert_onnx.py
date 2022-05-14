import argparse
import torch
import torch.nn as nn

import maskcyclegan.modules
import melgan.modules
import modules

class VC(nn.Module):
    def __init__(self, vc):
        super().__init__()
        self.vocoder = melgan.modules.Generator()
        self.vocoder.load_state_dict(torch.load("melgan/model.pt"))
        self.vc = vc
    
    # (batch_size, n_mel, length) -> (batch_size, wave_length)
    def forward(self, mel):
        mel_converted = self.vc(mel, torch.ones_like(mel))
        wav_converted = self.vocoder(mel_converted)
        return wav_converted


def main(args):
    model = modules.MaskCycleGAN_VC().load_from_checkpoint(args.model_path)
    if args.x2y:
        vc = VC(model.generator_XY)
    if args.y2x:
        vc = VC(model.generator_YX)
    model.to("cpu")
    model.eval()
    dummy_input = torch.randn(1, 80, 32, requires_grad=True)
    torch.onnx.export(
        vc,
        dummy_input,
        args.output_path,
        opset_version=13,
        input_names=["input_mel"],
        output_names=["output_wav"],
        dynamic_axes={
            "input_mel": {0: "batch_size", 2: "length"},
            "output_wav": {0: "batch_size", 1: "length"},
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--x2y", action="store_true")
    parser.add_argument("--y2x", action="store_true")
    args = parser.parse_args()

    assert args.x2y or args.y2x
    main(args)