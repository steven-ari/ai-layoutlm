import argparse
import os

import numpy as np
import torch
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import AdamW
from transformers import LayoutLMForTokenClassification
from transformers import LayoutLMTokenizer

from layoutlm_utils.sroie import SROIEDataset


# TODO change this into Pytorch Lightning, and this needs lots of refactoring

def dataloaders(tokenizer):
    args = {
        'local_rank': -1,
        'overwrite_cache': True,
        'data_dir': '/Users/steve/tasks/perga/ai/ai-layoutlm/datasets/label_studio/dummy_example/',
        'model_name_or_path': 'microsoft/layoutlm-base-uncased',
        'max_seq_length': 512,
        'model_type': 'layoutlm',
    }

    # Class to turn the keys of a dict into attributes
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self

    args = AttrDict(args)

    labels = ["S-COMPANY", "S-DATE", "S-ADDRESS", "S-TOTAL", "O"]
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # The LayoutLM authors already defined a specific FunsdDataset, so we are going to use this here
    train_dataset = SROIEDataset(args, tokenizer, labels, pad_token_label_id, mode="train")
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=2)

    eval_dataset = SROIEDataset(args, tokenizer, labels, pad_token_label_id, mode="test")
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=2)

    return train_dataloader, eval_dataloader


def train(model, train_dataloader, optimizer, device):
    batch = next(iter(train_dataloader))

    global_step = 0
    # put the model in training mode
    model.train()
    for batch in tqdm(train_dataloader, desc="Training"):
        input_ids = batch[0].to(device)
        bbox = batch[4].to(device)
        attention_mask = batch[1].to(device)
        token_type_ids = batch[2].to(device)
        labels = batch[3].to(device)
        # forward pass
        outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
                        labels=labels)
        loss = outputs.loss
        # print loss every 100 steps
        if global_step % 100 == 0:
            print(f"Loss after {global_step} steps: {loss.item()}")

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        global_step += 1


def test(model, device, eval_dataloader):
    labels = ["S-COMPANY", "S-DATE", "S-ADDRESS", "S-TOTAL", "O"]
    label_map = {i: label for i, label in enumerate(labels)}

    pad_token_label_id = CrossEntropyLoss().ignore_index
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    # put model in evaluation mode
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            input_ids = batch[0].to(device)
            bbox = batch[4].to(device)
            attention_mask = batch[1].to(device)
            token_type_ids = batch[2].to(device)
            labels = batch[3].to(device)

            # forward pass
            outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=labels)

            # get the loss and logits
            tmp_eval_loss = outputs.loss
            logits = outputs.logits
            eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1

            # compute the predictions
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, labels.detach().cpu().numpy(), axis=0
                )

    # compute average evaluation loss
    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }
    return results


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='LayoutLM training')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to parsed (default: 14)')
    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    use_mps = torch.backends.mps.is_available() and not args.no_mps

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
    model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased", num_labels=5)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Dataloader    
    train_dataloader, eval_dataloader = dataloaders(tokenizer)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(model, train_dataloader, optimizer, device)
        test(model, device, eval_dataloader)
        scheduler.step()

    if args.save_model:
        save_dir = "trained_layoutlm"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, "training_model.pt")
        torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    main()
