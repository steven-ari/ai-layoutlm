import json
import os
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import pytesseract
from PIL import Image
from tqdm import tqdm

from layoutlm_utils.bbox import normalize_bbox

# TODO needs refactoring, this file will take a lot of time

# create label.txt file
LABELS = ['COMPANY', 'DATE', 'ADDRESS', 'TOTAL']


def extract_labelstudio_labels(json_file, output_entities_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_entities_dir, exist_ok=True)

    # Load the JSON data
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Process each item in the JSON data
    for item in data:
        # Extract the file name from the "ocr" variable and remove file extension
        file_path = item["ocr"]
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        # Extract label.labels and transcription fields
        converted_data = {}
        labels = item["label"]
        transcriptions = item["transcription"]
        for label, transcription in zip(labels, transcriptions):
            converted_data[label["labels"][0]] = transcription

        # Generate the output file path
        output_file = os.path.join(output_entities_dir, f"{file_name}.txt")

        # Write the converted data to the output file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(converted_data, f, indent=4, ensure_ascii=False)


def extract_text_from_ocr(input_image_dir, output_box_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_box_dir, exist_ok=True)

    # Process each image file in the input directory
    for file_name in os.listdir(input_image_dir):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_image_dir, file_name)

            # Load the image
            image = Image.open(image_path)

            # Perform OCR using pytesseract
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

            # Extract the text and coordinates
            extracted_text = []
            for i, text in enumerate(ocr_data['text']):
                if text.strip():  # Ignore empty text
                    x0 = ocr_data['left'][i]
                    y0 = ocr_data['top'][i]
                    x1 = x0 + ocr_data['width'][i]
                    y1 = y0
                    x2 = x0 + ocr_data['width'][i]
                    y2 = y0 + ocr_data['height'][i]
                    x3 = x0
                    y3 = y0 + ocr_data['height'][i]
                    extracted_text.append((x0, y0, x1, y1, x2, y2, x3, y3, text))

            # Extract the image file name without extension
            file_name = os.path.splitext(os.path.basename(image_path))[0]

            # Generate the output file path
            output_path = os.path.join(output_box_dir, f"{file_name}.txt")

            # Save the extracted text and coordinates to the .txt file
            with open(output_path, 'w') as file:
                for item in extracted_text:
                    x0, y0, x1, y1, x2, y2, x3, y3, text = item
                    line = f"{x0},{y0},{x1},{y1},{x2},{y2},{x3},{y3},{text}\n"
                    file.write(line)


def create_labels_file(labels, output_dir, io_tags=['S'], non_labeled_tag='O'):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'labels.txt', 'w') as f:
        for tag in io_tags:
            for label in labels:
                f.write(f"{tag}-{label}\n")
        f.write(non_labeled_tag)


def read_bbox_and_words(path: Path):
    bbox_and_words_list = []

    with open(path, 'r', errors='ignore') as f:
        for line in f.read().splitlines():
            if len(line) == 0:
                continue

            split_lines = line.split(",")

            bbox = np.array(split_lines[0:8], dtype=np.int32)
            text = ",".join(split_lines[8:])

            # From the splited line we save (filename, [bounding box points], text line).
            # The filename will be useful in the future
            bbox_and_words_list.append([path.stem, *bbox, text])

        dataframe = pd.DataFrame(bbox_and_words_list,
                                 columns=['filename', 'x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'line'])

        # Drop columns that are not needed
        dataframe = dataframe.drop(columns=['x1', 'y1', 'x3', 'y3'])

        # Reorder the columns
        dataframe = dataframe[['filename', 'x0', 'y0', 'x2', 'y2', 'line']]

    return dataframe


def read_entities(path: Path):
    with open(path, 'r') as f:
        data = json.load(f)

    dataframe = pd.DataFrame([data])
    return dataframe


def assign_line_label(line: str, entities: pd.DataFrame):
    line_set = line.replace(",", "").strip().split()
    for i, column in enumerate(entities):
        entity_values = entities.iloc[0, i].replace(",", "").strip()
        entity_set = entity_values.split()

        matches_count = 0
        for l in line_set:
            if any(SequenceMatcher(a=l, b=b).ratio() > 0.8 for b in entity_set):
                matches_count += 1

            if (column.upper() == 'ADDRESS' and (matches_count / len(line_set)) >= 0.5) or \
                    (column.upper() != 'ADDRESS' and (matches_count == len(line_set))) or \
                    matches_count == len(entity_set):
                return column.upper()

    return "O"


def assign_labels(words: pd.DataFrame, entities: pd.DataFrame):
    max_area = {"TOTAL": (0, -1), "DATE": (0, -1)}  # Value, index
    already_labeled = {"TOTAL": False,
                       "DATE": False,
                       "ADDRESS": False,
                       "COMPANY": False,
                       "O": False
                       }

    # Go through every line in $words and assign it a label
    labels = []
    for i, line in enumerate(words['line']):
        label = assign_line_label(line, entities)

        already_labeled[label] = True
        if label == "ADDRESS" and (already_labeled["DATE"] or already_labeled["TOTAL"]):
            label = "O"

        labels.append(label)

    words["label"] = labels
    return words


def split_line(line: pd.Series):
    line_copy = line.copy()

    line_str = line_copy.loc["line"]
    words = line_str.split(" ")

    # Filter unwanted tokens
    words = [word for word in words if len(word) >= 1]

    x0, y0, x2, y2 = line_copy.loc[['x0', 'y0', 'x2', 'y2']]
    bbox_width = x2 - x0

    new_lines = []
    for index, word in enumerate(words):
        x2 = x0 + int(bbox_width * len(word) / len(line_str))

        line_copy.loc[['x0', 'x2', 'line']] = [x0, x2, word]
        new_lines.append(line_copy.to_list())
        x0 = x2 + 5

    return new_lines


def dataset_creator(folder: Path):
    bbox_folder = folder / 'box'
    entities_folder = folder / 'entities'
    img_folder = folder / 'img'

    # Sort by filename so that when zipping them together
    # we don't get some other file (just in case)
    entities_files = sorted(entities_folder.glob("*.txt"))
    bbox_files = sorted(bbox_folder.glob("*.txt"))
    img_files = sorted(img_folder.glob("*.jpg"))

    data = []

    print("Reading dataset:")
    for bbox_file, entities_file, img_file in tqdm(zip(bbox_files, entities_files, img_files), total=len(bbox_files)):
        # Read the files
        bbox = read_bbox_and_words(bbox_file)
        entities = read_entities(entities_file)
        image = Image.open(img_file)

        # Assign labels to lines in bbox using entities
        bbox_labeled = assign_labels(bbox, entities)
        del bbox

        # Split lines into separate tokens
        new_bbox_l = []
        for index, row in bbox_labeled.iterrows():
            new_bbox_l += split_line(row)
        new_bbox = pd.DataFrame(new_bbox_l, columns=bbox_labeled.columns, dtype=object)

        del bbox_labeled

        # Do another label assignment to keep the labeling more precise
        for index, row in new_bbox.iterrows():
            label = row['label']

            if label != "O":
                entity_values = entities.iloc[0, entities.columns.get_loc(label.lower())]
                entity_set = entity_values.split()

                if any(SequenceMatcher(a=row['line'], b=b).ratio() > 0.7 for b in entity_set):
                    label = "S-" + label
                else:
                    label = "O"

            new_bbox.at[index, 'label'] = label

        width, height = image.size

        data.append([new_bbox, width, height])

    return data


def write_dataset(dataset: list, output_dir: Path, name: str):
    print(f"Writing {name}ing dataset:")
    with open(output_dir / f"{name}.txt", "w+", encoding="utf8") as file, \
            open(output_dir / f"{name}_box.txt", "w+", encoding="utf8") as file_bbox, \
            open(output_dir / f"{name}_image.txt", "w+", encoding="utf8") as file_image:

        # Go through each dataset
        for datas in tqdm(dataset, total=len(dataset)):
            data, width, height = datas

            if data.empty:
                continue  # Skip the empty DataFrame and move to the next iteration

            filename = data.iloc[0, data.columns.get_loc('filename')]

            # Go through every row in dataset
            for index, row in data.iterrows():
                bbox = [int(p) for p in row[['x0', 'y0', 'x2', 'y2']]]
                normalized_bbox = normalize_bbox(bbox, height, width)

                file.write("{}\t{}\n".format(row['line'], row['label']))
                file_bbox.write("{}\t{} {} {} {}\n".format(row['line'], *normalized_bbox))
                file_image.write("{}\t{} {} {} {}\t{} {}\t{}\n".format(row['line'], *bbox, width, height, filename))

            # Write a second newline to separate dataset from others
            file.write("\n")
            file_bbox.write("\n")
            file_image.write("\n")

    create_labels_file(LABELS, output_dir)
