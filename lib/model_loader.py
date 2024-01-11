# coding=utf-8
#
# LICENSE OF USING THE FOLLOWING MODELS
#
# LLAMA 2 COMMUNITY LICENSE AGREEMENT: https://github.com/facebookresearch/llama/blob/main/LICENSE
# Mistral LICENSE: https://www.apache.org/licenses/LICENSE-2.0

import torch
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    IntervalStrategy,
)

from lib.evaluation import compute_metrics
from utils.utils import (
    add_special_token,
    print_parameters,
    tokenizer_embedding_resize,
)


def model_initialize(config):
    model_name = config.get("model_name")
    device_map = config.get("device_map")
    train_max_len = config.get("train_max_len")
    hf_auth_token = config.get("hf_auth_token")
    g_save_dir = config.get("g_save_dir")
    llama_path = config.get("llama_path")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        llama_path,
        use_auth_token=hf_auth_token,
        device_map=device_map,
        torch_dtype=torch.float16,
    )
    model.config.pad_token_id = model.config.eos_token_id

    print('Number of trainable parameters of the Generator G before adding LoRA!')
    print_parameters(model)
    print('\n')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=g_save_dir,
        model_max_length=train_max_len,
        add_eos_token=True,
        add_bos_token=True,
        padding='longest',
        padding_side="right",
        truncation=True,
        return_tensors="pt",
        use_fast=False,
        trust_remote_code=True,
        use_auth_token=hf_auth_token,
        device_map=device_map,
    )
    if tokenizer.pad_token is None:
        tokenizer_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            tokenizer=tokenizer,
            model=model,
        )
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer = add_special_token(tokenizer)

    # Resize input token embeddings matrix of the model if new_num_tokens != config.vocab_size.
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def trainer_loader(config, lora, model, tokenizer, data_module, num_train_epochs):
    lora_alpha = config.get("lora_alpha")
    lora_dropout = config.get("lora_dropout")
    lora_r = config.get("lora_r")
    train_batch_size = config.get("g_train_batch_size")
    eval_batch_size = config.get("g_eval_batch_size")
    gradient_accumulation_steps = config.get("gradient_accumulation_steps")
    optim = config.get("optim")
    logging_steps = config.get("logging_steps")
    learning_rate = config.get("learning_rate")
    weight_decay = config.get("weight_decay")
    warmup_ratio = config.get("warmup_ratio")
    lr_scheduler_type = config.get("lr_scheduler_type")
    fp16 = config.get("fp16")
    bf16 = config.get("bf16")
    g_save_dir = config.get("g_save_dir")

    if lora:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        model.add_adapter(lora_config, adapter_name="adapter")
        model.enable_adapters()

    arguments = TrainingArguments(
        output_dir=g_save_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_steps=-1,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        report_to="tensorboard",
    )

    # Set supervised fine-tuning parameters
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        **data_module,
        args=arguments,
    )

    print('Number of trainable parameters of the Generator G after adding LoRA!')
    print_parameters(model)
    print('\n')

    return trainer
