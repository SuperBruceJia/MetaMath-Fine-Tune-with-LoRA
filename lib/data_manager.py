# coding=utf-8

import gc
import re
import random
import copy
import multiprocessing
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Features, Value, concatenate_datasets
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor.adapters import lora
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

from utils.utils import (
    load_config,
    add_special_token,
    perturbation,
)

IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{question}\n\n### Response: Let's think step by step."
    ),
}

# Load configuration
config = load_config()
model_name = config.get("model_name")
device_map = config.get("device_map")
hf_auth_token = config.get("hf_auth_token")
g_save_dir = config.get("g_save_dir")
result_dir = config.get("result_dir")
train_max_len = config.get("train_max_len")
max_new_tokens = config.get("max_new_tokens")
train_path = config.get("train_path")
num_cpu_cores = config.get("num_cpu_cores")
num_gpus = config.get("num_gpus")
llama_path = config.get("llama_path")

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=result_dir,
    return_tensors="pt",
    model_max_length=train_max_len,
    add_eos_token=True,
    add_bos_token=True,
    padding='longest',
    padding_side="right",  # Padding 32,000 on the right side for input_ids
    use_fast=False,
    trust_remote_code=True,
    use_auth_token=hf_auth_token,
    device_map=device_map,
)
tokenizer = add_special_token(tokenizer)
tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default


def tokenize_func(examples):
    return tokenizer(examples["text"], padding=True, truncation=True)


def perturbation_worker(sentence, answer):
    return perturbation(sen=sentence, ratio=1.0, answer=answer)


def metamath_dataset_loader():
    """
    Dataset description
    Dataset({
        features: ['type', 'answer', 'question'],
        num_rows: 395000
    })
    """
    # Load the dataset
    dataset = load_dataset("meta-math/MetaMathQA")
    dataset = dataset["train"]

    # Change "query" to "question" and "response" to "answer"
    dataset = dataset.rename_column('query', 'question')
    dataset = dataset.rename_column('response', 'answer')

    return dataset


def _tokenize_fn(strings, tokenizer):
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_lens,
        labels_lens=labels_lens,
    )


def preprocess(sources, targets, tokenizer):
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]

    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)

    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self):
        super(SupervisedDataset, self).__init__()
        tokenizer_kwargs = {'padding': True, 'truncation': True, 'max_length': train_max_len}

        # Load prompt format
        prompt_format = PROMPT_DICT["prompt"]

        # Load Metamath dataset
        data_dict = metamath_dataset_loader()

        # Source and target
        sources = [prompt_format.format_map(example) for example in data_dict]
        targets = [f"{example['answer']}" for example in data_dict]
        self.sources = sources
        self.targets = targets

    def __len__(self):
        return len(self.sources)

    def naive__getitem__(self, i):
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i]
        )

    def __getitem__(self, i):
        return dict(
            input_ids=self.sources[i],
            labels=self.targets[i]
        )


@dataclass
class DataCollator(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def naive__call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

    def __call__(self, instances):
        sources = []
        targets = []

        for instance in instances:
            source = instance['input_ids']
            target = instance['labels']

            sources.append(source)
            targets.append(target)

        data = preprocess(sources, targets, self.tokenizer)
        input_ids, labels = data['input_ids'], data['labels']

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def dataset_loader(tokenizer, discriminator, pre_train=False):
    """Make dataset and collator for supervised fine-tuning."""
    dataset = SupervisedDataset()
    data_collator = DataCollator(tokenizer=tokenizer)

    return dict(
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=data_collator
    )
