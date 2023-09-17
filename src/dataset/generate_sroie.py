from pathlib import Path

from layoutlm_utils.dataset_utils import extract_text_from_ocr, dataset_creator, write_dataset, create_labels_file


def main():
    sroie_folder_path = Path('SROIE2019_without_box')

    # parsed dir path
    input_image_dir_train = "SROIE2019_without_box/parsed/img"
    output_box_dir_train = "SROIE2019_without_box/parsed/box"

    # test dir path
    input_image_dir_test = "SROIE2019_without_box/test/img"
    output_box_dir_test = "SROIE2019_without_box/test/box"

    # extract parsed dataset
    extract_text_from_ocr(input_image_dir_train, output_box_dir_train)
    # extract test dataset
    extract_text_from_ocr(input_image_dir_test, output_box_dir_test)

    # sroie dataset creator
    dataset_train = dataset_creator(sroie_folder_path / 'parsed')
    dataset_test = dataset_creator(sroie_folder_path / 'test')

    dataset_directory = Path('sroie_train_data')
    dataset_directory.mkdir(parents=True, exist_ok=True)

    # create label.txt file
    labels = ['COMPANY', 'DATE', 'ADDRESS', 'TOTAL']

    create_labels_file(labels, dataset_directory)

    # sroie dataset write
    write_dataset(dataset_train, dataset_directory, 'parsed')
    write_dataset(dataset_test, dataset_directory, 'test')


if __name__ == "__main__":
    main()
