import argparse
import importlib
import os
import subprocess
import tempfile
from faster_whisper import WhisperModel, available_models

# Max characters to send to the translation API at once
MAX_CHR = 1000
# Readability delay at onset of each segment (milliseconds)
DELAY_IN = 100
# Readability delay at offset of each segment (milliseconds)
DELAY_OUT = 500
# Minimal duration for a segment to remove artifcats (milliseconds)
MIN_DURATION = 300
# Heuristic repetitive sequence detection: number above which a sequence is removed
MAX_NUM_REPEATS = 3

def valid_lang(lang):
    return lang in googletrans.LANGUAGES.keys()


def is_audio_file(path):
    ext = os.path.splitext(path)[1]
    return ext in ['.mp3', '.wav', '.ogg', '.flac', '.m4a']


def is_subtitle_file(path):
    return os.path.splitext(path)[1] == '.srt'


def time_encode(seconds):
    hours, left = int(seconds // 3600), seconds % 3600
    minutes, left = int(left // 60), left % 60
    seconds, left = int(left // 1), left % 1
    return f'{hours:02d}:{minutes:02d}:{seconds:02d},{int(left * 1000):03d}'


def time_decode(time):
    hours, minutes, seconds = time.split(':')
    seconds, milliseconds = seconds.split(',')
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000


def extract_audio(input, audio_file):
    print('Reading from', input)
    res = subprocess.run(['ffmpeg', '-i', input, '-vn', '-sn', '-c:a', 'libmp3lame', audio_file])
    if res.returncode != 0:
        raise Exception('ffmpeg failed')
    else:
        print('Audio extracted to', audio_file)
    return


def detect_repeats(seq, seed_len=4):
    # This function detects some artifacts produced by whisper in the form of 
    # repetitive sequences, like this one this one this one this one this on
    for i in range(len(seq) - seed_len):
        seed = seq[i:i+seed_len]
        for j in range(i+seed_len, len(seq)-seed_len):
            if seq[j:j+seed_len] == seed:
                repeat = seq[i:j]
                all_repeats = True
                k = j
                counts = 0
                while all_repeats and k < len(seq):
                    if k + len(repeat) > len(seq):
                        break
                    if seq[k:k+len(repeat)] != repeat:
                        all_repeats = False
                    k += len(repeat)
                    counts += 1
                if all_repeats and counts > MAX_NUM_REPEATS:
                    print(f'Warning: repetitive sequence with repeat {repeat} in sequence {seq}')
                    return True


def segment_filters(segment):
    # Remove empty segments
    if segment.text.strip() == '':
        return False
    
    # Remove artifact mini segments
    if segment.end - segment.start < MIN_DURATION / 1000:
        return False
    
    # Remove repetitive segments
    if detect_repeats(segment.text):
        return False
    
    return True


def postprocess_srt(text):
    # Here we adjust the ends of segments to avoid overlapping
    segments = text.split('\n\n')
    for i in range(len(segments) - 2):
        start, end = segments[i].split('\n')[0].split(' --> ')
        end = time_decode(end)
        next_start, _ = segments[i+1].split('\n')[0].split(' --> ')
        next_start = time_decode(next_start)
        if next_start < end:
            new_end = time_encode(next_start - DELAY_IN / 1000)
            new_timings = f'{start} --> {new_end}'
            segments[i] = '\n'.join([new_timings] + segments[i].split('\n')[1:])
    return '\n\n'.join(segments)
    
    
def transcribe_audio(audio, model_size, device='auto', compute_type='default', audio_lang=None):
    print(f'Loading model {model_size}')
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    print('Starting transcription')
    
    if audio_lang is None:
        segments, info = model.transcribe(audio, word_timestamps=True, vad_filter=True)
        print(f'Detected language: {info.language} with probability {info.language_probability}')
    else:
        if audio_lang not in model.supported_languages:
            raise Exception(f'Language {audio_lang} not supported by model {model_size}')
        print(f'Transcribing with forced language {audio_lang}')
        segments, info = model.transcribe(audio, word_timestamps=True, vad_filter=True, language=audio_lang,
                                          condition_on_previous_text=False)
        
    # Convert to SRT format
    text = ''
    for segment in segments:
        if not segment_filters(segment):
            continue
        start = time_encode(segment.start + DELAY_IN / 1000)
        end = time_encode(segment.end + DELAY_OUT / 1000)
        text += f'{start} --> {end}\n'
        text += segment.text + '\n\n'
        print(f'{start} --> {end} : {segment.text}\r', end='')
        
    text = postprocess_srt(text)
    return text, info.language


def batch_together(segments, max_chr):
    batches = []
    cur_batch = ''
    for segment in segments:
        if len(cur_batch) + len(segment) + 2 > max_chr and cur_batch != '':
            batches.append(cur_batch)
            cur_batch = ''
        cur_batch += segment.strip() + '\n\n'
    if cur_batch != '':
        batches.append(cur_batch)
    return batches


def translate(text, src, dest, superpose=False, color='yellow'):
    translator = googletrans.Translator()
    print('Translating to', dest)
    chunks = text.split('\n\n')
    segments = ['\n'.join(chunk.split('\n')[1:]) for chunk in chunks]
    if src is None:
        src = 'auto'
    
    # Translate all segments, aggregated in batches (to keep max translation coherence while staying
    # within Google's limits)
    translation_batches = batch_together(segments, MAX_CHR)
    translated_batches = [translator.translate(batch, src=src, dest=dest) for batch in translation_batches]
    translated_segments = []
    for batch in translated_batches:
        translated_segments.extend(batch.text.strip().split('\n\n'))
    
    print(len(segments))
    print(len(translated_segments))
    output = ''
    for i, segment in enumerate(segments):
        times = chunks[i].split('\n')[0]
        translated = translated_segments[i] if i < len(translated_segments) else '...'
        if superpose:
            output += times + '\n' \
                      + segment + '\n' \
                      + f' <font color="{color}">' + translated + '</font>\n\n'
        else:
            output += times + '\n' + translated + '\n\n'
    return output              
    
    
def main():
    print('Subtitlr v0.1')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input video or audio or subtitle file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output subtitle file')
    parser.add_argument('--model-size', type=str, default='tiny',
                        choices=available_models(),
                        help='Model size (see openai/whisper repo)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu, cuda, cuda:0, ...)')
    parser.add_argument('--compute-type', type=str, default='default',
                        choices=['int8', 'float16', 'float32'],
                        help='Compute type (default, float16, float32, float64)')
    parser.add_argument('--translate', type=str, default=None,
                        help='If set, translate to this language (e.g. en, fr, es, ...)')
    parser.add_argument('--superpose', action='store_true', default=False,
                        help='If set, superpose the original and translated subtitles')
    parser.add_argument('--audio-lang', type=str, default=None,
                        help='If set, force the original language (e.g. en, fr, es, ...)')
    args = parser.parse_args()
    
    if args.translate is not None:
        global googletrans
        googletrans = importlib.import_module('googletrans')
        if not valid_lang(args.translate):
            raise ValueError("Invalid language code {args.translate}")
    
    if not is_subtitle_file(args.input):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Audio extraction
            if not is_audio_file(args.input):
                audio_file = os.path.join(tmpdir, 'audio.mp3')
                extract_audio(args.input, audio_file)
            else:
                audio_file = args.input
                
            # Audio transcription
            text, lang = transcribe_audio(
                audio_file, 
                args.model_size, 
                device=args.device,
                compute_type=args.compute_type,
                audio_lang=args.audio_lang
            )
    # Translation
    if args.translate is not None:
        if is_subtitle_file(args.input):
            with open(args.input, 'r') as f:
                text = f.read()
            lang = args.audio_lang
        if lang == args.translate:
            print('Source and destination languages are the same. Skipping translation')
        else:
            text = translate(text, lang, args.translate, args.superpose)
        
        
    if args.output is None:
        output_file = os.path.splitext(args.input)[0] + '.srt'
    else:
        output_file = args.output
        
    with open(output_file, 'w') as f:
        f.write(text)
    print(f'Subtitles saved to {output_file}')
    print('Done')
    

if __name__ == '__main__':
    main()