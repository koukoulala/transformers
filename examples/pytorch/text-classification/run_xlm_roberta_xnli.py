#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning multi-lingual models on XNLI (e.g. Bert, DistilBERT, XLM).
    Adapted from `examples/text-classification/run_glue.py`"""

import argparse
import logging
import math
import os
import random

import datasets
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.utils.versions import require_version

logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="The name of the language to train on."
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--cache_dir", type=str, default=None, help="Where to store the ckpt downloaded from huggingface.co",
    )
    parser.add_argument(
        "--data_cache_dir", type=str, default=None, help="Where to store the datasets downloaded from huggingface.co",
    )
    parser.add_argument(
        "--do_train",  action="store_true", help="If passed, do_train",
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="If passed, do_eval",
    )
    parser.add_argument(
        "--do_predict", action="store_true", help="If passed, do_predict",
    )
    args = parser.parse_args()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args

def evaluated(model, args, eval_dataloader, accelerator, metric):
    model.eval()
    for step, batch in enumerate(eval_dataloader):
        #print("batch: ", batch)
        outputs = model(**batch)
        #print("outputs.logits: ", outputs.logits)
        predictions = outputs.logits.argmax(dim=-1)
        metric.add_batch(
            predictions=accelerator.gather(predictions),
            references=accelerator.gather(batch["labels"]),
        )

    eval_metric = metric.compute()
    logger.info(f"results: {eval_metric}")

    return model, eval_metric, predictions


def main():
    # See all possible arguments in src/transformers/args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # Downloading and loading xnli dataset from the hub.
    if args.do_train:
        train_dataset = load_dataset("xnli", args.language, split="train", cache_dir=args.data_cache_dir)
        label_list = train_dataset.features["label"].names

    if args.do_eval:
        eval_dataset = load_dataset("xnli", args.language, split="validation", cache_dir=args.data_cache_dir)
        label_list = eval_dataset.features["label"].names

    if args.do_predict:
        predict_dataset = load_dataset("xnli", args.language, split="test", cache_dir=args.data_cache_dir)
        label_list = predict_dataset.features["label"].names

    # Labels
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, cache_dir=args.cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, cache_dir=args.cache_dir)

    model = accelerator.prepare(model)

    # Preprocessing the datasets
    # Padding strategy
    padding = "max_length" if args.pad_to_max_length else False

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    def preprocess_function(examples):
        # Tokenize the texts
        examples["hypothesis"] = ["This example is " + x + "." for x in examples["hypothesis"]]
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding=padding,
            max_length=args.max_length,
            truncation=True,
            truncation_strategy='only_first'
        )

    if args.do_train:
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            #remove_columns=train_dataset.column_names,
            desc="Running tokenizer on train dataset",
        )
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
        )

    if args.do_eval:
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            #remove_columns=eval_dataset.column_names,
            desc="Running tokenizer on validation dataset",
        )
        for index in random.sample(range(len(eval_dataset)), 3):
            logger.info(f"Sample {index} of the eval_dataset: {eval_dataset[index]}.")
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
        eval_dataloader = accelerator.prepare(eval_dataloader)

    if args.do_predict:
        predict_dataset = predict_dataset.map(
            preprocess_function,
            batched=True,
            #remove_columns=predict_dataset.column_names,
            desc="Running tokenizer on prediction dataset",
        )
        for index in random.sample(range(len(predict_dataset)), 3):
            logger.info(f"Sample {index} of the predict_dataset: {predict_dataset[index]}.")

        predict_dataloader = DataLoader(predict_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
        predict_dataloader = accelerator.prepare(predict_dataloader)

    # Get the metric function
    metric = load_metric("accuracy")

    if args.do_train:
        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        # Prepare everything with our `accelerator`.
        optimizer, train_dataloader = accelerator.prepare(optimizer, train_dataloader)

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

        # Train!
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0

        for epoch in range(args.num_train_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                if completed_steps >= args.max_train_steps:
                    break
            if args.output_dir is not None:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)

    if args.do_eval:
        logger.info("doing evaluating: ")
        model, eval_metric, predictions = evaluated(model, args, eval_dataloader, accelerator, metric)

    if args.do_predict:
        logger.info("doing predicting: ")
        model, predict_metric, predictions = evaluated(model, args, predict_dataloader, accelerator, metric)
        output_predict_file = os.path.join(args.output_dir, "predictions.txt")
        with open(output_predict_file, "w") as writer:
            writer.write("index\tprediction\n")
            for index, item in enumerate(predictions):
                item = label_list[item]
                writer.write(f"{index}\t{item}\n")


if __name__ == "__main__":
    main()
