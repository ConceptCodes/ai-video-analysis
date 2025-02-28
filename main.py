import argparse

from utils import (
    split_video_into_chunks,
    create_assets_dir,
    analyse_chunk,
    extract_audio_from_video,
    transcribe_audio,
)
from constants import VIDEO_CHUNK_LENGTH


def setup_cli():
    parser = argparse.ArgumentParser(
        description="This is simple AI program to perform video verification"
    )
    parser.add_argument(
        "-v",
        "--video",
        type=str,
        default="",
        help="Path to the video file",
    )
    parser.add_argument(
        "-d",
        "--description",
        type=str,
        default="",
        help="Path to the video description file",
    )

    args = parser.parse_args()

    if not args.video:
        parser.error("Please provide the path to the video file")

    if not args.description:
        parser.error("Please provide the path to the video description file")

    return args


def main():
    create_assets_dir()
    args = setup_cli()
    results = []

    chunks = split_video_into_chunks(args.video, VIDEO_CHUNK_LENGTH)

    for chunk in chunks:
        audio = extract_audio_from_video(chunk)
        transcription = transcribe_audio(audio)
        video_analysis = analyse_chunk(chunk)
        results.append(
            {
                "audio": transcription,
                "video": video_analysis,
            }
        )

    print(len(results))
