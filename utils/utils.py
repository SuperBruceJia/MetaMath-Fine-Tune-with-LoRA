# coding=utf-8

import io
import os
import re
import json
import yaml
import random

from data_processing.paragraph_split import paragraph_splitter
from data_augmentation.character import CharacterPerturb
from data_augmentation.word import WordPerturb
from data_augmentation.sentence import SentencePerturb

DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_UNK_TOKEN = "<unk>"


def tokenizer_embedding_resize(special_tokens_dict, tokenizer, model):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def add_special_token(tokenizer):
    tokenizer.add_special_tokens(
        {
            "pad_token": DEFAULT_PAD_TOKEN,
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        }
    )

    return tokenizer


def stop_token_list():
    stop_tokens = ["Question:",
                   "Question",
                   "USER:",
                   "USER",
                   "ASSISTANT:",
                   "ASSISTANT",
                   "Instruction:",
                   "Instruction",
                   "Response:",
                   "Response",]

    return stop_tokens


def gpu_usage():
    """
    Check the usage of GPU
    """
    os.system('nvidia-smi')


def load_config():
    """Load parameters and path from the YAML file

    :return: The configuration info
    """
    fopen = open("config.yml")
    config = yaml.load(fopen, Loader=yaml.FullLoader)
    fopen.close()

    return config


def print_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    # Retrieve a list of all named parameters in the model
    model_parameters = list(model.named_parameters())

    # Calculate the total number of parameters using a generator expression
    all_param = sum(p.numel() for _, p in model_parameters)

    # Calculate the total number of trainable parameters using a generator expression
    # that filters parameters which require gradients
    trainable_params = sum(p.numel() for _, p in model_parameters if p.requires_grad)

    # Print out the number of trainable parameters, total parameters,
    # and the percentage of parameters that are trainable
    # The percentage is formatted to two decimal places
    print(
        f"Trainable params: {trainable_params:,} | "
        f"All params: {all_param:,} | "
        f"Trainable%: {100 * trainable_params / all_param:.2f}%"
    )


class CustomStream:
    def __init__(self, filename, console_stream):
        self.filename = filename
        self.console_stream = console_stream

    def write(self, text):
        with open(self.filename, 'a') as file:
            file.write(text)
        self.console_stream.write(text)

    def flush(self):
        pass


def gsm8k_format(question, answer=None, inference=False):
    """The formatting prompts function for GSM8K database

    :param question: Question (task description)
    :param answer: Answer to the Question
    :return: The prompt of the GSM8K database
    """
    question = re.sub('[\n\t]', ' ', question)
    question = question.replace('  ', ' ')

    prompt = (f'{"Below is an instruction that describes a task. "}'
              f'{"Write a response that appropriately completes the request."}'
              + "\n\n### Instruction:\n" + question
              + "\n\n### Response: Let's think step by step.")

    if inference is False:
        answer = re.sub('[\n\t]', ' ', answer)
        answer = answer.replace('####', 'The answer is:')
        prompt += answer
    else:
        pass

    return prompt


def model_saver(trainer, output_dir):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict

        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def unwrap_model(model):
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


def perturbation(sen, ratio, answer):
    if random.random() >= ratio:
        pass
    else:
        if answer:
            sen_out = []
            sens = paragraph_splitter(paragraph=sen)
            for i in range(len(sens) - 1):
                sen = sens[i]
                sentence_tool = SentencePerturb(sentence=sen)
                level = random.sample(["bt_hugging_face",
                                       "bt_google",
                                       "formal",
                                       "casual",
                                       "passive",
                                       "active",
                                       "paraphrase"], 1)[0]

                # Sentence-level
                if level == "bt_hugging_face":
                    sen = sentence_tool.back_translation_hugging_face()
                elif level == "bt_google":
                    sen = sentence_tool.back_translation_google()
                elif level == "formal":
                    sen = sentence_tool.formal()
                elif level == "casual":
                    sen = sentence_tool.casual()
                elif level == "passive":
                    sen = sentence_tool.passive()
                elif level == "active":
                    sen = sentence_tool.active()
                else:
                    sen = sentence_tool.paraphrase()

                while sen is None:
                    level = random.sample(["bt_hugging_face",
                                           "bt_google",
                                           "formal",
                                           "casual",
                                           "passive",
                                           "active",
                                           "paraphrase"], 1)[0]
                    if level == "bt_hugging_face":
                        sen = sentence_tool.back_translation_hugging_face()
                    elif level == "bt_google":
                        sen = sentence_tool.back_translation_google()
                    elif level == "formal":
                        sen = sentence_tool.formal()
                    elif level == "casual":
                        sen = sentence_tool.casual()
                    elif level == "passive":
                        sen = sentence_tool.passive()
                    elif level == "active":
                        sen = sentence_tool.active()
                    else:
                        sen = sentence_tool.paraphrase()

                sen_out.append(sen)

            sen_out.append(sens[-1])

        else:
            sen_out = []
            sens = paragraph_splitter(paragraph=sen)
            for i in range(len(sens) - 1):
                sen = sens[i]
                level = random.sample(["char_replace",
                                       "char_delete",
                                       "char_insert",
                                       "char_swap",
                                       "char_keyboard",
                                       "char_ocr",
                                       "word_replace",
                                       "word_delete",
                                       "word_insert",
                                       "word_swap",
                                       "word_split",
                                       "word_punctuation",
                                       "bt_hugging_face",
                                       "bt_google",
                                       "formal",
                                       "casual",
                                       "passive",
                                       "active",
                                       "paraphrase"], 1)[0]

                noise_ratio = random.sample([0.05, 0.10, 0.15, 0.20, 0.25], 1)[0]
                character_tool = CharacterPerturb(sentence=sen, level=noise_ratio)
                word_tool = WordPerturb(sentence=sen, level=noise_ratio)
                sentence_tool = SentencePerturb(sentence=sen)

                if level == "char_replace":
                    sen = character_tool.character_replacement()
                elif level == "char_delete":
                    sen = character_tool.character_deletion()
                elif level == "char_insert":
                    sen = character_tool.character_insertion()
                elif level == "char_swap":
                    sen = character_tool.character_swap()
                elif level == "char_keyboard":
                    sen = character_tool.keyboard_typos()
                elif level == "char_ocr":
                    sen = character_tool.optical_character_recognition()
                elif level == "word_replace":
                    sen = word_tool.synonym_replacement()
                elif level == "word_delete":
                    sen = word_tool.word_deletion()
                elif level == "word_insert":
                    sen = word_tool.word_insertion()
                elif level == "word_swap":
                    sen = word_tool.word_swap()
                elif level == "word_split":
                    sen = word_tool.word_split()
                elif level == "word_punctuation":
                    sen = word_tool.insert_punctuation()
                elif level == "bt_hugging_face":
                    sen = sentence_tool.back_translation_hugging_face()
                elif level == "bt_google":
                    sen = sentence_tool.back_translation_google()
                elif level == "formal":
                    sen = sentence_tool.formal()
                elif level == "casual":
                    sen = sentence_tool.casual()
                elif level == "passive":
                    sen = sentence_tool.passive()
                elif level == "active":
                    sen = sentence_tool.active()
                else:
                    sen = sentence_tool.paraphrase()

                while sen is None:
                    # level = random.sample(["bt_hugging_face", "bt_google"], 1)[0]
                    level = random.sample(["bt_hugging_face",
                                           "bt_google",
                                           "formal",
                                           "casual",
                                           "passive",
                                           "active",
                                           "paraphrase"], 1)[0]

                    if level == "bt_hugging_face":
                        sen = sentence_tool.back_translation_hugging_face()
                    elif level == "bt_google":
                        sen = sentence_tool.back_translation_google()
                    elif level == "formal":
                        sen = sentence_tool.formal()
                    elif level == "casual":
                        sen = sentence_tool.casual()
                    elif level == "passive":
                        sen = sentence_tool.passive()
                    elif level == "active":
                        sen = sentence_tool.active()
                    else:
                        sen = sentence_tool.paraphrase()

                sen_out.append(sen)

            sen_out.append(sens[-1])

        if len(sen_out) > 1:
            sen = ' '.join(sen_out)
        else:
            sen = sen_out[0]

    return sen


def get_sys_input(query):
    if query.find('\n') == -1:
        return ''
    return '\n'.join(query.split('\n')[1:])
