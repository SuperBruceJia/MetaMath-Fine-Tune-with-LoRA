# coding=utf-8

import gc
import re
import sys
import time

import torch
import jsonlines
import numpy as np
from fraction import Fraction
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from vllm import LLM, SamplingParams
from vllm.model_executor.adapters import lora
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

from utils.utils import gsm8k_format, stop_token_list


MAX_INT = sys.maxsize


def compute_metrics(p):
    """
    Credits: https://stackoverflow.com/questions/69087044/early-stopping-in-bert-trainer-instances
    https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    """
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def is_number(s):
    """
    Codes Credits: https://github.com/meta-math/MetaMath/blob/main/eval_gsm8k.py
    :param s:
    :return:
    """
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def extract_number(completion):
    """
    Codes Credits: https://github.com/meta-math/MetaMath/blob/main/eval_gsm8k.py
    :param completion: The model's generated response
    :return: The extracted answer number from the completion
    """
    text = completion.split('The answer is:')

    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)

        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]

                if is_number(denominator) and is_number(numerator):
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None

            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))

        else:
            return None

    else:
        return None


def gsm8k_test(config, file_path, data_path):
    """
    Codes Credits: https://github.com/meta-math/MetaMath/blob/main/eval_gsm8k.py
    :param data_path: dataset path
    :param file_path: save file path and file name
    """
    start_t = time.time()
    max_new_tokens = config.get("max_new_tokens")
    # model_name = config.get("model_name")
    g_save_dir = config.get("g_save_dir")
    num_gpus = config.get("num_gpus")
    llama_path = config.get("llama_path")

    # Read the database and retrieve the label `gsm8k_answers`
    instances = []
    answers = []
    with open(data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            # Get the prompt template + question --> gsm8k_ins
            temp_ins = gsm8k_format(question=item["question"], answer=None, inference=True)
            instances.append(temp_ins)

            # Get the label answer --> gsm8k_answers
            temp_ans = item['answer'].split('#### ')[1]
            temp_ans = int(temp_ans.replace(',', ''))
            answers.append(temp_ans)

    generations = []
    stop_tokens = stop_token_list()
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=max_new_tokens, stop=stop_tokens)
    llm = LLM(model=llama_path, tensor_parallel_size=num_gpus, gpu_memory_utilization=0.90)
    lora.LoRAModel.from_pretrained(llm.llm_engine.workers[0].model, g_save_dir + '/adapter')

    completions = llm.generate(instances, sampling_params)
    for output in completions:
        temp_gen = output.outputs[0].text
        generations.append(temp_gen)

    print('Successfully finished generating', len(instances), 'samples!')
    result = []
    invalid_out = []
    for idx, (instance_item, generation_item, answer_item) in enumerate(zip(instances, generations, answers)):
        y_pred = extract_number(generation_item)
        if y_pred is not None:
            result.append(float(y_pred) == float(answer_item))
        else:
            result.append(False)

            temp = {'question': instance_item, 'output': generation_item, 'answer': answer_item}
            invalid_out.append(temp)

    accuracy = sum(result) / len(result)
    end_t = time.time()
    elapsed_t = end_t - start_t
    print(f"Finished performance evaluation in {elapsed_t:.2f} seconds")

    # Print the accuracy and the length of the invalid output
    print('Invalid output length:', len(invalid_out), ', Testing length:', len(result), ', Accuracy:', accuracy)

    # Save the invalid output in a txt file
    file = open(file_path, 'w')
    file.write(str(invalid_out))
    file.close()
    print('Successfully saved the invalid output.')

    # Delete the llm object and free the memory
    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.destroy_process_group()
    print("Successfully delete the llm pipeline and free the GPU memory.\n\n\n\n")
