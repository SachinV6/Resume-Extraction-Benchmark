import argparse
import os
import shutil

from tests.qualitative_benchmark import run_benchmark, DEFAULT_MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--output", default="outputs/qualitative_benchmark")
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    if args.clear:
        if os.path.exists(args.output):
            shutil.rmtree(args.output)

    os.makedirs(args.output, exist_ok=True)

    run_benchmark(pdf_path=args.pdf, models=DEFAULT_MODELS, output_dir=args.output)


if __name__ == "__main__":
    main()