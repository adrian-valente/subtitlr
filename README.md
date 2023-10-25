# subtitlr
Automatic subtitle generation &amp; translation in python with [faster-whisper](https://github.com/guillaumekln/faster-whisper) and [googletrans](https://github.com/ssut/py-googletrans).

Implicitly, ffmpeg is used for audio extraction (but required anyway by faster-whisper). Bearing in mind that googletrans uses an unofficial API, I am planning to integrate the [translate](https://github.com/terryyin/translate-python) module as an option soon (using the official APIs of deepl, Azure...).

## Installing
Via pip: create a virtual environment and run:
```sh
pip install git+https://github.com/adrian-valente/subtitlr/
```
From source: in the repository root folder, create a virtual environment and run:
```sh
python setup.py install
```
Note: to use CUDA, first install the relevant libraries:
```sh
conda install -y -c pytorch -c nvidia pytorch==1.10.2 torchvision torchaudio cudatoolkit=11.3  # Adapt to your CUDA version
conda install -y -c conda-forge cudnn
```

## Usage
A typical example run would be:
```sh
subtitlr -i video.mkv -o subs.srt --model-size large-v2 --device cuda --translate en --superpose
```
One can also directly provide an audio file:
```sh
subtitlr -i recoding.mp3 -o subs.srt --model-size large-v2 --device cuda --translate en --superpose
```
or even a subtitle file for simply translation:
```sh
subtitlr -i original.srt -o translated.srt --translate en --superpose
```
If doing the audio transcription, in languages other than english only the largest model sizes will work and a GPU might be necessary. It is also best to specify the original language with the option `--audio-lang` to avoid relying on automatic detection.

Complete user guide:
```sh
usage: subtitlr [-h] --input INPUT [--output OUTPUT]
                [--model-size {tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large}]
                [--device DEVICE] [--compute-type {int8,float16,float32}]
                [--translate TRANSLATE] [--superpose] [--audio-lang AUDIO_LANG]

options:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        Input video or audio or subtitle file
  --output OUTPUT, -o OUTPUT
                        Output subtitle file
  --model-size {tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large}
                        Model size (see openai/whisper repo)
  --device DEVICE       Device to use (cpu, cuda, cuda:0, ...)
  --compute-type {int8,float16,float32}
                        Compute type (default, float16, float32, float64)
  --translate TRANSLATE
                        If set, translate to this language (e.g. en, fr, es, ...)
  --superpose           If set, superpose the original and translated subtitles
  --audio-lang AUDIO_LANG
                        If set, force the original language (e.g. en, fr, es, ...)
```
