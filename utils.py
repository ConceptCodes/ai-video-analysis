import os
import whisper
import cv2
import base64
import io
import tempfile
from moviepy.editor import VideoFileClip
from langchain.schema import HumanMessage, SystemMessage

from llm import vision_llm

model = whisper.load_model("turbo")


def delete_directory_contents(directory):
    """
    Deletes all files and subdirectories in the specified directory.

    Args:
      directory (str): Path to the directory to be emptied.
    """
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            os.rmdir(item_path)


def split_video_into_chunks(video_path, chunk_duration):
    """
    Splits the input video into multiple chunks of specified duration.

    Args:
      video_path (str): Path to the input video file.
      chunk_duration (float): Duration (in seconds) for each chunk.

    Returns:
      list: List of file paths for the generated video chunks.
    """
    output_dir = "assets/chunks"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
    return chunks


def create_assets_dir():
    if not os.path.exists("assets"):
        os.makedirs("assets")


def transcribe_audio(audio_path):
    """
    Transcribes the input audio using the Whisper API.

    Args:
      audio (str): Path to the audio file to be transcribed.

    Returns:
      str: Transcribed text from the audio.
    """
    result = model.transcribe(audio_path)
    return result["text"]


def get_base64_image(frame):
    """
    Converts the input frame to a base64 encoded image.

    Args:
      frame (numpy.ndarray): Input frame to be converted.

    Returns:
      str: Base64 encoded image.
    """
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")


def analyse_chunk(chunk_path):
    """
    Analyzes the input video by transcribing its audio and extracting frames.

    Args:
      video_path (str): Path to the input video file.

    Returns:
      str: Transcribed text from the video audio.
    """
    cap = cv2.VideoCapture(chunk_path)
    frame_count = 0
    frames = []
    results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1

    cap.release()

    for frame in frames:
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
                    {"type": "text", "text": "What can you see in this image?"},
                    {
                        "type": "image_url",
                        "image_url": get_base64_image(frame),
                    },
                ]
            ),
        ]
        result = vision_llm(messages)
        results.append(result)

    return results


def extract_audio_from_video(video_path):
    """
    Extracts audio from the input video file and returns it as an in-memory WAV audio buffer.

    Args:
      video_path (str): Path to the input video file.

    Returns:
      io.BytesIO: In-memory buffer containing the WAV audio data.
    """
    video = VideoFileClip(video_path)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
        temp_audio_path = temp_file.name

    video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
    video.close()

    with open(temp_audio_path, "rb") as f:
        audio_data = f.read()
    os.remove(temp_audio_path)

    return io.BytesIO(audio_data)
