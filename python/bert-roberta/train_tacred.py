import argparse
import os

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from utils import set_seed, collate_fn
from prepro import TACREDProcessor
from evaluation import get_f1, compute_f1, compute_h_f1, accuracy, positive_accuracy, hierarchical_accuracy
from model import REModel
from torch.cuda.amp import GradScaler
from hierarchy_utils import constant
# import wandb

def train(args, model, train_features, benchmarks):
    train_dataloader = DataLoader(train_features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    total_steps = int(len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    scaler = GradScaler()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    print('Total steps: {}'.format(total_steps))
    print('Warmup steps: {}'.format(warmup_steps))

    num_steps = 0
    for epoch in range(int(args.num_train_epochs)):
        print("Epoch  ::  {}".format(epoch))
        model.zero_grad()
        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            inputs = {'input_ids': batch[0].to(args.device),
                      'attention_mask': batch[1].to(args.device),
                      'labels': batch[2].to(args.device),
                      'ss': batch[3].to(args.device),
                      'os': batch[4].to(args.device),
                      }
            outputs = model(**inputs)
            loss = outputs[0] / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if step % args.gradient_accumulation_steps == 0:
                num_steps += 1
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                model.zero_grad()
                # wandb.log({'loss': loss.item()}, step=num_steps)
                # print("Loss : {}", loss.item())

            if (num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                for tag, features in benchmarks:
                    if tag == 'dev':
                        f1, output = evaluate(args, model, features, tag=tag)
    

    for tag, features in benchmarks:
        f1, output = evaluate(args, model, features, tag=tag)
        # wandb.log(output, step=num_steps)


def evaluate(args, model, features, tag='dev'):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, collate_fn=collate_fn, drop_last=False)
    keys, preds = [], []
    for i_b, batch in enumerate(dataloader):
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'ss': batch[3].to(args.device),
                  'os': batch[4].to(args.device),
                  }
        keys += batch[2].tolist()
        with torch.no_grad():
            logit = model(**inputs)[0]
            pred = torch.argmax(logit, dim=-1)
        preds += pred.tolist()

    keys = np.array(keys, dtype=np.int64)
    preds = np.array(preds, dtype=np.int64)
    p, r, max_f1 = get_f1(keys, preds)
    if tag == 'test' or tag == 'test_rev':
        print(tag.upper()+" EVALUATION")
        compute_f1(preds, keys)
        compute_h_f1(preds, keys)
        accuracy(preds, keys)
        positive_accuracy(preds, keys)
        hierarchical_accuracy(preds, keys)

    output = {
        tag + "_p": p * 100,
        tag + "_r": r * 100,
        tag + "_f1": max_f1 * 100,
    }
    print(output)
    return max_f1, output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./data/tacred", type=str)
    parser.add_argument("--model_name_or_path", default="roberta-large", type=str)
    parser.add_argument("--input_format", default="typed_entity_marker_punct", type=str,
                        help="in [entity_mask, entity_marker, entity_marker_punct, typed_entity_marker, typed_entity_marker_punct]")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated.")

    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=32, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int,
                        help="Number of updates steps to accumulate the gradients for, before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=5.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=42)
    parser.add_argument("--evaluation_steps", type=int, default=500,
                        help="Number of steps to evaluate the model")

    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--project_name", type=str, default="RE_baseline")
    parser.add_argument("--run_name", type=str, default="tacred")

    parser.add_argument('--filter_out', action='store_true', 
                        help='Remove setences with ambiguous subject entity type from data.')
    parser.add_argument('--relabel', action='store_true', 
                        help='Hierarchical Re-labelling: Mapping finer positive relations to coarser relation from hierarchy.')

    parser.add_argument('--hier_loss', action='store_true',
                        help="Whether to use hierarchical CrossEntropyLoss or normal CrossEntropyLoss")
    parser.add_argument('--lca', action='store_true', 
                        help='True uses lca based distance for hierarchical distance loss, False uses simple distance.')
    parser.add_argument('--hier_normalise', action='store_true', 
                        help="Normalise Hierarchical Distance loss")
    parser.add_argument('--hier_log', action='store_true', 
                        help="Log of Hierarchical Distance loss")
    parser.add_argument('--all_pos', action='store_true', help='Trains a classifier for all positive relations.')

    args = parser.parse_args()
    # wandb.init(project=args.project_name, name=args.run_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    if args.seed > 0:
        set_seed(args)

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    config.gradient_checkpointing = True
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    if args.hier_loss:
        print("Loss Function  ::  Hierarchical CrossEntropyLoss")
        if args.hier_normalise:
            print("Normalisation Distance {}".format(args.hier_normalise))
        if args.hier_log:
            print("Logarithm Distance {}".format(args.hier_log))
    else:
        print("Loss Function  ::  CrossEntropyLoss")
    print("")

    model = REModel(args, config)
    model.to(0)

    if args.relabel:
        constant.LABEL_TO_ID = constant.RELABELED_LABEL_TO_ID
    if args.all_pos:
        print('Considering only positive labeled sentences..!')
        constant.LABEL_TO_ID.pop('no_relation')
        constant.LABEL_TO_ID = {key:val-1 for key, val in constant.LABEL_TO_ID.items()}

    train_file = os.path.join(args.data_dir, "train.json")
    dev_file = os.path.join(args.data_dir, "dev.json")
    test_file = os.path.join(args.data_dir, "test.json")
    dev_rev_file = os.path.join(args.data_dir, "dev_rev.json")
    test_rev_file = os.path.join(args.data_dir, "test_rev.json")

    processor = TACREDProcessor(args, tokenizer)
    train_features = processor.read(train_file, args.filter_out, args.relabel, args.all_pos)
    dev_features = processor.read(dev_file, args.filter_out, args.relabel, args.all_pos)
    test_features = processor.read(test_file, args.filter_out, args.relabel, args.all_pos)
    dev_rev_features = processor.read(dev_rev_file, args.filter_out, args.relabel, args.all_pos)
    test_rev_features = processor.read(test_rev_file, args.filter_out, args.relabel, args.all_pos)

    if len(processor.new_tokens) > 0:
        model.encoder.resize_token_embeddings(len(tokenizer))

    benchmarks = (
        ("dev", dev_features),
        ("test", test_features),
        ("dev_rev", dev_rev_features),
        ("test_rev", test_rev_features),
    )

    train(args, model, train_features, benchmarks)


if __name__ == "__main__":
    main()
