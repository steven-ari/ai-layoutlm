from pathlib import Path

from layoutlm_utils.dataset_utils import extract_labelstudio_labels, extract_text_from_ocr, dataset_creator, \
    write_dataset


def main():
    # extract plain SROIE format from Label Studio, also use pytesseract for OCR
    ls_dataset_dir = Path('/Users/steve/tasks/perga/ai/ai-layoutlm/datasets/label_studio/dummy_example')
    ls_dataset_train_dir = ls_dataset_dir / 'labelstudio_result/parsed'
    ls_dataset_json = ls_dataset_dir / 'dummy_export.json'

    input_image_dir = ls_dataset_train_dir / "img"
    output_entities_dir = ls_dataset_train_dir / "entities"
    output_box_dir = ls_dataset_train_dir / "box"

    extract_labelstudio_labels(ls_dataset_json, output_entities_dir)
    extract_text_from_ocr(input_image_dir, output_box_dir)

    # convert into a directory format that is necessary for the dataset class
    dataset_ls = dataset_creator(ls_dataset_train_dir)
    dataset_directory = ls_dataset_dir / 'train'
    dataset_directory.mkdir(parents=True, exist_ok=True)
    write_dataset(dataset_ls, dataset_directory, 'train')


if __name__ == "__main__":
    main()
