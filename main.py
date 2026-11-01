"""RetailPulse command-line entry point.

    python main.py                    # generate data + run everything
    python main.py --skip-generate    # use your own CSVs already placed in data/raw/
"""
import argparse
import logging

from retailpulse import pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="RetailPulse end-to-end analytics pipeline")
    parser.add_argument("--skip-generate", action="store_true",
                        help="do not generate synthetic data; read the CSVs already in data/raw/")
    parser.add_argument("-v", "--verbose", action="store_true", help="show data-quality details")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    if not args.verbose:
        logging.getLogger("retailpulse.pipeline").setLevel(logging.INFO)
    pipeline.run(generate=not args.skip_generate)
    print("\nFinished. Open outputs/retail_report.xlsx and run:  streamlit run app.py")


if __name__ == "__main__":
    main()
