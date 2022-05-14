
# What is this?
Unofficial implement of [MaskCycleGAN-VC](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/maskcyclegan-vc/index.html).

# Usage

1. Collect your dataset and put your dataset into `datasets` directory.

example
```
datasets
|-- VoiceKugimiya
    |-- louise.wav
    ...
|-- VoiceMe
    |-- hogefuga.wav
    ...
```

2. install requirements

```
$ pip install -r requirements.txt
```

3. Training your model

Your model is saved into `models` directory.

```
$ python train.py --dataset_x datasets/VoiceMe --dataset_y datasets/VoiceKugimiya --sample_x datasets/VoiceMe/hogefuga.wav
```

4. Test your model
```
$ python test.py --model_path models/mask_cyclegan_vc.ckpt --wav_path your_voice.wav --output_path converted_your_voice.wav --x2y
```
