import os
import gradio as gr
from transformers import pipeline
from pytube import YouTube
from datasets import Dataset, Audio
from moviepy.editor import AudioFileClip
from deep_translator import GoogleTranslator

pipe = pipeline(model="Neprox/model")

languages = [
    "English (en)",
    "German (de)",
    "French (fr)",
    "Spanish (es)",
]

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

def get_translation(text, target_lang="English (en)"):
    """
    Translates the given Swedish text to the language specified.
    """
    lang_code = target_lang.split(" ")[-1][1:-1]
    return GoogleTranslator(source='sv', target=lang_code).translate(text)

def translate(audio, url, seconds_max, target_lang):
    """
    Translates a YouTube video if a url is specified and returns the transcription.
    If not url is specified, it translates the audio file as passed by Gradio.
    :param audio: Audio file as passed by Gradio. Only used if no url is specified.
    :param url: URL of the YouTube video to translate.
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
            text += f"[Translation to {target_lang}]\n"
            text += f"{get_translation(output['text'], target_lang)}\n\n"
        return text

    else:
        text = pipe(audio)["text"]
    return text

iface = gr.Interface(
    fn=translate, 
    inputs=[
        gr.Audio(source="microphone", type="filepath", label="Translate from Microphone"),
        gr.Text(max_lines=1, placeholder="Enter YouTube Link with Swedish speech to be translated", label="Translate from YouTube URL"),
        gr.Slider(minimum=30, maximum=300, value=30, step=30, label="Number of seconds to translate from YouTube URL"),
        gr.Dropdown(languages, value="English (en)", label="Target language")
    ], 
    outputs="text",
    title="Whisper Small Swedish",
    description="Realtime demo for Swedish speech recognition with translation using a fine-tuned Whisper small model.",
)

iface.launch()
