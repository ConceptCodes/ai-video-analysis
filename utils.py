import os
import whisper
import cv2
import base64
import io
import tempfile
import numpy as np
import torch
from halo import Halo
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from langchain.schema import HumanMessage, SystemMessage

from llm import vision_llm, base_llm

model = whisper.load_model("turbo")


def delete_directory_contents(dir_path: str):
    """
    Deletes all files and subdirectories in the specified directory.

    Args:
      dir_path (str): Path to the directory to be emptied.
    """
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory {dir_path} does not exist.")

    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            os.rmdir(item_path)


def split_video_into_chunks(video_path: str, chunk_duration: float):
    """
    Splits the input video into multiple chunks of specified duration.

    Args:
      video_path (str): Path to the input video file.
      chunk_duration (float): Duration (in seconds) for each chunk.

    Returns:
      list: List of file paths for the generated video chunks.
    """
    spinner = Halo(text="Splitting video into chunks", spinner="dots")
    spinner.start()

    output_dir = "assets/chunks"
    video = VideoFileClip(video_path)
    total_duration = video.duration
    chunks = []
    start = 0
    index = 0

    while start < total_duration:
        end = min(start + chunk_duration, total_duration)
        subclip = video.subclip(start, end)
        chunk_file = os.path.join(output_dir, f"chunk_{index}.mp4")

        subclip.write_videofile(
            chunk_file, codec="libx264", audio_codec="aac", verbose=False, logger=None
        )
        chunks.append(chunk_file)

        start += chunk_duration
        index += 1

    video.close()
    spinner.succeed(f"Generated {len(chunks)} video chunks.")
    return chunks


def create_assets_dir():
    if not os.path.exists("assets"):
        os.makedirs("assets")
    if not os.path.exists("assets/chunks"):
        os.makedirs("assets/chunks")


def transcribe_audio(audio_buffer: io.BytesIO):
    """
    Transcribes the input audio using the Whisper API.

    Args:
      audio_buffer (io.BytesIO): In-memory buffer containing the audio data.

    Returns:
      str: Transcribed text from the audio.
    """
    spinner = Halo(text="Transcribing audio", spinner="dots")
    spinner.start()

    audio_buffer.seek(0)
    audio_data = np.frombuffer(audio_buffer.read(), np.int16)
    # Normalize the audio data
    audio_tensor = torch.from_numpy(audio_data).float() / 32768.0
    result = model.transcribe(audio_tensor)

    spinner.succeed("Transcription complete")
    return result["text"]


def get_base64_image(frame: np.ndarray):
    """
    Converts the input frame to a base64 encoded image.

    Args:
      frame (numpy.ndarray): Input frame to be converted.

    Returns:
      str: Base64 encoded image.
    """
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")


def analyse_chunk(chunk_path: str):
    """
    Analyzes the input video by analyzing each frame using a vision model.

    Args:
      chunk_path (str): Path to the input video file.

    Returns:
      str: Transcribed text from the video audio.
    """
    spinner = Halo(text="Analyzing video chunk", spinner="dots")
    spinner.start()

    cap = cv2.VideoCapture(chunk_path)
    frame_count = 0
    frames = []
    results = []
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1

    cap.release()
    spinner.succeed(f"Extracted {frame_count} frames from the video chunk.")

    frames = [frame for i, frame in enumerate(frames) if i % int(fps) == 0]

    for frame in tqdm(frames, desc="Analyzing video frames"):
        messages = [
            SystemMessage(
                content=[
                    {
                        "type": "text",
                        "text": "I analyse images and provide detailed description of the content.",
                    }
                ]
            ),
            HumanMessage(
                content=[
                    {"type": "text", "text": "What can you see in this frame?"},
                    {
                        "type": "image_url",
                        "image_url": get_base64_image(frame),
                    },
                ]
            ),
        ]
        result = vision_llm(messages)
        results.append(result.content)

    spinner = Halo(text="Summarizing video chunk", spinner="dots")
    spinner.start()
    messages = [
        SystemMessage(
            content=[
                {
                    "type": "text",
                    "text": "I summarize the content of the video segment based on the analysis of individual frames. Please be as detailed as possible.",
                }
            ]
        ),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "What is the overall theme of the video segment?",
                },
                {"type": "text", "text": "\n".join(results)},
            ]
        ),
    ]
    chunk_summary = base_llm(messages)

    spinner.succeed("Video chunk analysis complete.")
    return chunk_summary.content


def extract_audio_from_video(video_path: str):
    """
    Extracts audio from the input video file and returns it as an in-memory WAV audio buffer.

    Args:
      video_path (str): Path to the input video file.

    Returns:
      io.BytesIO: In-memory buffer containing the WAV audio data.
    """
    spinner = Halo(text="Extracting audio from video", spinner="dots")
    spinner.start()
    video = VideoFileClip(video_path)

    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
        temp_audio_path = temp_file.name

    video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
    video.close()

    with open(temp_audio_path, "rb") as f:
        audio_data = f.read()
    os.remove(temp_audio_path)

    spinner.succeed("Audio extraction complete.")
    return io.BytesIO(audio_data)


def generate_report(requirements: str, results: list):
    with open(requirements, "r") as f:
        requirements = f.read()
    messages = [
        SystemMessage(
            content=[
                {
                    "type": "text",
                    "text": "Your goal is to analyze the video segments and determine if they meet the specified requirements.",
                }
            ]
        ),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Based on the following requirements and the analysis of the video segments, generate a report.",
                },
                {"type": "text", "text": "Requirements:"},
                {"type": "text", "text": requirements},
                {"type": "text", "text": "Results:"},
                {"type": "text", "text": "\n".join(results)},
            ]
        ),
    ]
    result = base_llm(messages)
    return result.content
