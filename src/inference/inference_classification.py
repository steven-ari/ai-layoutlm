import argparse

import cv2
import torch
from transformers import LayoutLMForSequenceClassification

from layoutlm_utils.image import apply_ocr_to_image
from layoutlm_utils.utils import encode_sample

DOC_LABELS = ['email', 'invoice', 'resume', 'scientific_publication']


def main(main_arg: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LayoutLMForSequenceClassification.from_pretrained("microsoft/layoutlm-base-uncased")
    model.eval()
    model.to(device)

    page_img = cv2.imread(main_arg.input_image_file)
    ocr_result = apply_ocr_to_image(page_img)
    encoded_input = encode_sample(ocr_result)

    with torch.no_grad():
        outputs = model(
            input_ids=torch.tensor([encoded_input.data['input_ids']]).to(device),
            attention_mask=torch.tensor([encoded_input.data['attention_mask']]).to(device),
            token_type_ids=torch.tensor([encoded_input.data['token_type_ids']]).to(device),
            bbox=torch.tensor([encoded_input.data['bbox']]).to(device),
        )

        preds = torch.softmax(outputs.logits, dim=1).tolist()[0]  # get output of first item in batch

    pred_labels = {label: pred for label, pred in zip(DOC_LABELS, preds)}
    category_prediction = max(pred_labels, key=pred_labels.get)
    print(pred_labels)
    print(f'document prediction: {category_prediction}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-image-file', type=str, required=True)
    main_args = parser.parse_args()
    main(main_args)
