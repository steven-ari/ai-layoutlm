import argparse

from layoutlm_utils.pdf import extract_metadata_from_pdf


def main(args):
    metadata = extract_metadata_from_pdf(args.input_pdf)
    print(metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-pdf', type=str, required=True)
    main_args = parser.parse_args()
    main(main_args)
