import argparse
from PIL import ImageDraw, ImageFont
import torch
from transformers import LayoutLMForTokenClassification

from layoutlm_utils.utils import preprocess, extract_predictions, LABELS_NER

PATH = '/Users/steve/tasks/perga/ai/ai-layoutlm/training_model.pt'

labels = LABELS_NER
num_labels = len(labels)
label_map = {i: label for i, label in enumerate(labels)}


def main(main_arg: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased",
                                                           num_labels=num_labels)
    model.load_state_dict(torch.load(PATH, map_location=device))
    model.to(device)
    model.eval()

    image, words, boxes, actual_boxes = preprocess(main_arg.input_image_file)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    word_level_predictions, final_boxes = extract_predictions(image, words,
                                                              boxes, actual_boxes, model, device)
    label2color = {'S-ADDRESS': 'red',
                   'S-DATE': 'blue',
                   'S-TOTAL': 'green',
                   'S-COMPANY': 'blue',
                   'O': 'grey'}
    for prediction, box in zip(word_level_predictions, final_boxes):
        if prediction >= num_labels:
            continue

        predicted_label = label_map[prediction]
        draw.rectangle(box, outline=label2color[predicted_label], width=1)
        draw.text((box[0] + 10, box[1] - 10),
                  text=predicted_label,
                  fill=label2color[predicted_label],
                  font=font)

    image.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-image-file', type=str, required=True)
    main_args = parser.parse_args()
    main(main_args)
