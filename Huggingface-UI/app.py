import os
import gradio as gr
from transformers import pipeline
from pytube import YouTube
from datasets import Dataset, Audio
from moviepy.editor import AudioFileClip

pipe = pipeline(model="Neprox/model")

def download_from_youtube(url):
    """
    Downloads the video from the given YouTube URL and returns the path to the audio file.
    """
    streams = YouTube(url).streams.filter(only_audio=True, file_extension='mp4')
    fpath = streams.first().download()
    return fpath

def get_timestamp(seconds):
    """
    Creates %M:%S timestamp from seconds.
    """
    minutes = int(seconds / 60)
    seconds = int(seconds % 60)
    return f"{str(minutes).zfill(2)}:{str(seconds).zfill(2)}"

def divide_into_30s_segments(audio_fpath, seconds_max):
    """
    Divides the audio file into 30s segments and returns the paths to the segments and the start times of the segments.
    :param audio_fpath: Path to the audio file.
    :param seconds_max: Maximum number of seconds to consider. If the audio file is longer than this, it will be truncated.
    """
    if not os.path.exists("segmented_audios"):
        os.makedirs("segmented_audios")

    sound = AudioFileClip(audio_fpath)
    n_full_segments = int(sound.duration / 30)
    len_last_segment = sound.duration % 30

    max_segments = int(seconds_max / 30)
    if n_full_segments > max_segments:
        n_full_segments = max_segments
        len_last_segment = 0

    segment_paths = []
    segment_start_times = []

    segments_available = n_full_segments + 1
    for i in range(min(segments_available, max_segments)):
        start = i * 30

        # Skip last segment if it is smaller than two seconds
        is_last_segment = i == n_full_segments
        if is_last_segment and not len_last_segment > 2:
            continue
        elif is_last_segment:
            end = start + len_last_segment
        else:
            end = (i + 1) * 30

        segment_path = os.path.join("segmented_audios", f"segment_{i}.wav")
        segment = sound.subclip(start, end)
        segment.write_audiofile(segment_path)
        segment_paths.append(segment_path)
        segment_start_times.append(start)

    return segment_paths, segment_start_times

def get_translation(text):
    """
    Translates the given Swedish text to English.
    """
    # TODO: Make API call to Google Translate to get English translation
    return "..."

def transcribe(audio, url, seconds_max):
    """
    Transcribes a YouTube video if a url is specified and returns the transcription.
    If not url is specified, it transcribes the audio file as passed by Gradio.
    :param audio: Audio file as passed by Gradio. Only used if no url is specified.
    :param url: YouTube URL to transcribe.
    :param seconds_max: Maximum number of seconds to consider. If the audio file is longer than this, it will be truncated.
    """
    if url:
        fpath = download_from_youtube(url)
        segment_paths, segment_start_times = divide_into_30s_segments(fpath, seconds_max)

        audio_dataset = Dataset.from_dict({"audio": segment_paths}).cast_column("audio", Audio(sampling_rate=16000))
        pred = pipe(audio_dataset["audio"])
        text = ""
        n_segments = len(segment_start_times)
        for i, (seconds, output) in enumerate(zip(segment_start_times, pred)):
            text += f"[Segment {i+1}/{n_segments}, start time {get_timestamp(seconds)}]\n"
            text += f"{output['text']}\n"
            text += f"[Translation]\n{get_translation(output['text'])}\n\n"
        return text

    else:
        text = pipe(audio)["text"]
    return text

iface = gr.Interface(
    fn=transcribe, 
    inputs=[
        gr.Audio(source="microphone", type="filepath", label="Transcribe from Microphone"),
        gr.Text(max_lines=1, placeholder="Enter YouTube Link with Swedish speech to be transcribed", label="Transcribe from YouTube URL"),
        gr.Slider(minimum=30, maximum=300, value=30, step=30, label="Number of seconds to transcribe from YouTube URL")
    ], 
    outputs="text",
    title="Whisper Small Swedish",
    description="Realtime demo for Swedish speech recognition using a fine-tuned Whisper small model.",
)

iface.launch()
