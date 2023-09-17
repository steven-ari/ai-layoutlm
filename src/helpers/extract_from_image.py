from pathlib import Path

from layoutlm_utils.image import extract_metadata_from_images


def main(images_paths: list[Path]):
    metadata = extract_metadata_from_images(images_paths)
    print(metadata)


if "__main__" == __name__:
    images_paths = [
        Path(
            "/label_studio/dummy_example/labelstudio_result/parsed/img/91c642d0-invoice_image.jpg"),
        Path(
            "/label_studio/dummy_example/labelstudio_result/parsed/img/744dad0e-page_12.jpg")]
    main(images_paths)
