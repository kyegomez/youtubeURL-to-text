# 🎉 youtubeURL-to-text 🚀

Transform YouTube URLs into text 📝 100x faster 🏎️ with whisperx 🔥

Easily convert any YouTube video 🎥 into text using the power of whisperx 🌠. Perfect for transcription, subtitles, or any other text-based applications!

## 📦 Installation

```bash
!pip install pytube
!pip install pydub
!pip install git+https://github.com/m-bain/whisperx.git@v3
!pip install torch torchvision torchaudio
# !pip install pyannote.audio
```


## Permissions
You need Huggingface api key at: https://huggingface.co/settings/tokens

And huggingface will ask you to agree to terms of some models so be observant!

## 🚀 Usage

```python
import os
from pydub import AudioSegment
from pytube import YouTube
import whisperx

def download_youtube_video(video_url, audio_format='mp3'):
    audio_file = f'video.{audio_format}'
    
    # Download video 📥
    yt = YouTube(video_url)
    yt_stream = yt.streams.filter(only_audio=True).first()
    yt_stream.download(filename='video.mp4')

    # Convert video to audio 🎧
    video = AudioSegment.from_file("video.mp4", format="mp4")
    video.export(audio_file, format=audio_format)
    os.remove("video.mp4")
    
    return audio_file

def transcribe_youtube_video(video_url):
    audio_file = download_youtube_video(video_url)
    
    device = "cuda"
    batch_size = 16
    compute_type = "float16"

    # 1. Transcribe with original Whisper (batched) 🗣️
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)

    # 2. Align Whisper output 🔍
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # 3. Assign speaker labels 🏷️
    diarize_model = whisperx.DiarizationPipeline(use_auth_token='hugging face stable api key', device=device)
    diarize_segments = diarize_model(audio_file)
    
    try:
      segments = result["segments"]
      transcription = " ".join(segment['text'] for segment in segments)
      # return segments
      return transcription
    except KeyError:
      print("The key 'segments' is not found in the result.")

# Example usage
video_url = "url"
transcription = transcribe_youtube_video(video_url)
print(transcription)
```

## 🌟 Features

- Download YouTube videos as audio 🎵
- Lightning-fast transcription using whisperx 🌩️
- Output aligned with original audio 🎞️
- Speaker labeling to differentiate speakers in the transcription 🎤

Embrace the power of whisperx and revolutionize your YouTube transcription experience! 🚀
