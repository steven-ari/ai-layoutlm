from typing import Optional

import numpy as np
import pytesseract
import torch
from PIL import Image
from transformers import LayoutLMForTokenClassification
from transformers import LayoutLMTokenizer

from layoutlm_utils.bbox import normalize_bbox

labels = ['email', 'invoice', 'resume', 'scientific_publication']
idx2label = {v: k for v, k in enumerate(labels)}
label2idx = {k: v for v, k in enumerate(labels)}
tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")

LABELS_NER = ["S-COMPANY", "S-DATE", "S-ADDRESS", "S-TOTAL", "O"]


# TODO refactoring, this one will take time

def encode_sample(sample: dict, max_seq_length: int = 512, pad_token_box: Optional[list] = None):
    if pad_token_box is None:
        pad_token_box = [0, 0, 0, 0]
    words = sample['words']
    normalized_word_boxes = sample['bbox']
    assert len(words) == len(normalized_word_boxes)

    token_boxes = []
    for word, box in zip(words, normalized_word_boxes):
        word_tokens = tokenizer.tokenize(word)
        token_boxes.extend([box] * len(word_tokens))

    # Truncation of token_boxes
    special_tokens_count = 2
    if len(token_boxes) > max_seq_length - special_tokens_count:
        token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]

    # add bounding boxes of cls + sep tokens
    token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

    # Padding of token_boxes up the bounding boxes to the sequence length.
    encoding = tokenizer(' '.join(words), padding='max_length', truncation=True)
    input_ids = tokenizer(' '.join(words), truncation=True)["input_ids"]
    padding_length = max_seq_length - len(input_ids)
    token_boxes += [pad_token_box] * padding_length
    encoding['bbox'] = token_boxes

    assert len(encoding['input_ids']) == \
           len(encoding['attention_mask']) == \
           len(encoding['token_type_ids']) == \
           len(encoding['bbox']) == max_seq_length

    return encoding


def preprocess(image_path):
    image = Image.open(image_path)
    image = image.convert("RGB")

    width, height = image.size
    w_scale = 1000 / width
    h_scale = 1000 / height
    ocr_df = pytesseract.image_to_data(image, output_type='data.frame')
    ocr_df = ocr_df.dropna().assign(left_scaled=ocr_df.left * w_scale,
                                    width_scaled=ocr_df.width * w_scale,
                                    top_scaled=ocr_df.top * h_scale,
                                    height_scaled=ocr_df.height * h_scale,
                                    right_scaled=lambda x: x.left_scaled + x.width_scaled,
                                    bottom_scaled=lambda x: x.top_scaled + x.height_scaled)
    float_cols = ocr_df.select_dtypes('float').columns
    ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
    ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
    ocr_df = ocr_df.dropna().reset_index(drop=True)

    words = list(ocr_df.text)
    coordinates = ocr_df[['left', 'top', 'width', 'height']]
    actual_boxes = []
    for idx, row in coordinates.iterrows():
        x, y, w, h = tuple(row)  # the row comes in (left, top, width, height) format
        actual_box = [x, y, x + w, y + h]  # we turn it into (left, top, left+widght, top+height) to get the actual box
        actual_boxes.append(actual_box)
    boxes = []
    for box in actual_boxes:
        boxes.append(normalize_bbox(box, width, height))
    return image, words, boxes, actual_boxes


def convert_to_features(image, words, boxes, actual_boxes, tokenizer, args, cls_token_box=[0, 0, 0, 0],
                        sep_token_box=[1000, 1000, 1000, 1000], pad_token_box=[0, 0, 0, 0]):
    """
    Convert image, words, and bounding boxes to input features for a model using a tokenizer.

    Args:
        image (PIL.Image.Image): The input image.
        words (List[str]): List of words corresponding to the image.
        boxes (List[List[int]]): List of bounding boxes for each word in the image.
        actual_boxes (List[List[int]]): List of actual bounding boxes for each word in the image.
        tokenizer (Tokenizer): Tokenizer object for converting words to tokens.
        args: Additional arguments.
        cls_token_box (List[int], optional): Bounding box coordinates for the [CLS] token. Defaults to [0, 0, 0, 0].
        sep_token_box (List[int], optional): Bounding box coordinates for the [SEP] token. Defaults to [1000, 1000, 1000, 1000].
        pad_token_box (List[int], optional): Bounding box coordinates for padding tokens. Defaults to [0, 0, 0, 0].

    Returns:
        Tuple: Tuple of input features including input_ids, input_mask, segment_ids, token_boxes, and token_actual_boxes.

    Raises:
        AssertionError: If the lengths of the generated input features do not match args.max_seq_length.
    """

    width, height = image.size
    tokens = []
    token_boxes = []
    actual_bboxes = []
    token_actual_boxes = []
    for word, box, actual_bbox in zip(words, boxes, actual_boxes):
        word_tokens = tokenizer.tokenize(word)
        tokens.extend(word_tokens)
        token_boxes.extend([box] * len(word_tokens))
        actual_bboxes.extend([actual_bbox] * len(word_tokens))
        token_actual_boxes.extend([actual_bbox] * len(word_tokens))
    # Truncation: account for [CLS] and [SEP] with "- 2".
    special_tokens_count = 2
    if len(tokens) > args['max_seq_length'] - special_tokens_count:
        tokens = tokens[: (args['max_seq_length'] - special_tokens_count)]
        token_boxes = token_boxes[: (args['max_seq_length'] - special_tokens_count)]
        actual_bboxes = actual_bboxes[: (args['max_seq_length'] - special_tokens_count)]
        token_actual_boxes = token_actual_boxes[: (args['max_seq_length'] - special_tokens_count)]
    # add [SEP] token, with corresponding token boxes and actual boxes
    tokens += [tokenizer.sep_token]
    token_boxes += [sep_token_box]
    actual_bboxes += [[0, 0, width, height]]
    token_actual_boxes += [[0, 0, width, height]]

    segment_ids = [0] * len(tokens)
    # next: [CLS] token
    tokens = [tokenizer.cls_token] + tokens
    token_boxes = [cls_token_box] + token_boxes
    actual_bboxes = [[0, 0, width, height]] + actual_bboxes
    token_actual_boxes = [[0, 0, width, height]] + token_actual_boxes
    segment_ids = [1] + segment_ids
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)
    # Zero-pad up to the sequence length.
    padding_length = args['max_seq_length'] - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    input_mask += [0] * padding_length
    segment_ids += [tokenizer.pad_token_id] * padding_length
    token_boxes += [pad_token_box] * padding_length
    token_actual_boxes += [pad_token_box] * padding_length
    assert len(input_ids) == args['max_seq_length']
    assert len(input_mask) == args['max_seq_length']
    assert len(segment_ids) == args['max_seq_length']
    assert len(token_boxes) == args['max_seq_length']
    assert len(token_actual_boxes) == args['max_seq_length']

    return input_ids, input_mask, segment_ids, token_boxes, token_actual_boxes


def extract_predictions(image, words, boxes, actual_boxes, model, device):
    """
    Converts image, words and boxes into features that can be used as input to a LayoutLM model.

    Args:
        image (PIL.Image): The image to extract features from.
        words (list): A list of words in the image.
        boxes (list): A list of normalized bounding boxes for each word in the image.
        actual_boxes (list): A list of unnormalized bounding boxes for each word in the image.
        model (LayoutLMForTokenClassification): A LayoutLM model to use for feature extraction.

    Returns:
        tuple: A tuple containing:
            word_level_predictions (list): A list of predicted labels for each word in the image.
            final_boxes (list): A list of unnormalized bounding boxes for each word in the image.
    """
    args = {'local_rank': -1,
            'overwrite_cache': True,
            'model_name_or_path': 'microsoft/layoutlm-base-uncased',
            'max_seq_length': 512,
            'model_type': 'layoutlm'}

    input_ids, input_mask, segment_ids, token_boxes, token_actual_boxes = convert_to_features(image=image,
                                                                                              args=args)
    input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
    attention_mask = torch.tensor(input_mask, device=device).unsqueeze(0)
    token_type_ids = torch.tensor(segment_ids, device=device).unsqueeze(0)
    bbox = torch.tensor(token_boxes, device=device).unsqueeze(0)
    outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids)
    token_predictions = outputs.logits.argmax(-1).squeeze().tolist()  # the predictions are at the token level

    word_level_predictions = []  # let's turn them into word level predictions
    final_boxes = []
    for id, token_pred, box in zip(input_ids.squeeze().tolist(), token_predictions, token_actual_boxes):
        if (tokenizer.decode([id]).startswith("##")) or (id in [tokenizer.cls_token_id,
                                                                tokenizer.sep_token_id,
                                                                tokenizer.pad_token_id]):
            continue
        else:
            word_level_predictions.append(token_pred)
            final_boxes.append(box)
    return word_level_predictions, final_boxes
