import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import re

from utils import (
    split_video_into_chunks,
    create_assets_dir,
    analyse_chunk,
    extract_audio_from_video,
    transcribe_audio,
    delete_directory_contents,
)
from constants import VIDEO_CHUNK_LENGTH, NUM_WORKERS

warnings.filterwarnings(
    "ignore",
    message=re.escape(
        "A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy"
    ),
    category=UserWarning,
    module="scipy",
)

warnings.filterwarnings("ignore")


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

    args = parser.parse_args()

    if not args.video:
        parser.error("Please provide the path to the video file")

    return args


def process_chunk(chunk):
    audio = extract_audio_from_video(chunk)
    transcription = transcribe_audio(audio)
    video_analysis = analyse_chunk(chunk)
    return {
        "text": transcription,
        "video": video_analysis,
    }


def main():
    create_assets_dir()
    args = setup_cli()
    results = []

    chunks = split_video_into_chunks(args.video, VIDEO_CHUNK_LENGTH)

    # with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    #     futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
    #     for future in as_completed(futures):
    #         results.append(future.result())

    for chunk in chunks:
        result = process_chunk(chunk)
        results.append(result)

    delete_directory_contents("assets/chunks")

    print(len(results))


if __name__ == "__main__":
    main()
