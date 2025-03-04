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
    generate_report,
    is_filepath_valid,
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
        required=True,
    )
    parser.add_argument(
        "--txt",
        type=str,
        default="",
        help="Path to the text file",
        required=True,
    )

    args = parser.parse_args()

    if not is_filepath_valid(args.video):
        parser.error("Please provide the path to the video file")

    if not is_filepath_valid(args.txt):
        parser.error("Please provide the path to the requirements text file")

    return args


def process_chunk(chunk):
    audio = extract_audio_from_video(chunk)
    transcription = transcribe_audio(audio)
    video_analysis = analyse_chunk(chunk)
    return {
        "transcription": transcription,
        "video_analysis": video_analysis,
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

    for index, chunk in enumerate(chunks):
        print(f"\nProcessing chunk {index + 1} of {len(chunks)}")
        result = process_chunk(chunk)
        results.append(result)

    report = generate_report(args.txt, results)
    print(report)

    delete_directory_contents("assets/chunks")


if __name__ == "__main__":
    main()
