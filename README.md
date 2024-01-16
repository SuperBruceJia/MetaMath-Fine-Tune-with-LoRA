# Fine-tune LLaMA 2 (7B) with LoRA on meta-math/MetaMathQA
Fine-tuning and Inference codes on the MetaMath Dataset

[![Code License](https://img.shields.io/badge/Code%20License-MIT-green.svg)](CODE_LICENSE)
[![License](https://img.shields.io/badge/Running%20on-GPU-red.svg)](https://github.com/SuperBruceJia/MetaMath-Fine-Tune-with-LoRA)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

**P.S.:** Please reach out to [Shuyue Jia](https://github.com/SuperBruceJia) if you would be interested in supporting compute need. We are in need of some small-scale assistance at the moment, so any kind of help would be greatly appreciated. Thank you very much in advance!

## Model Details
`MetaMath-Fine-Tune-with-LoRA` is trained to reason and answer mathematical problems on [`meta-math/MetaMathQA`](https://huggingface.co/datasets/meta-math/MetaMathQA) dataset. We used [`meta-llama/Llama-2-7b-hf`](https://huggingface.co/meta-llama/Llama-2-7b-hf) as a base model and used **LoRA to fine-tune it**.

### Model Description
- **Project GitHub Page:** [https://github.com/SuperBruceJia/MetaMath-Fine-Tune-with-LoRA](https://github.com/SuperBruceJia/MetaMath-Fine-Tune-with-LoRA)
- **Developed by:** [Shuyue Jia](https://shuyuej.com/) in December, 2023
- **Funded by:** Boston University SCC
- **Model type:** fine-tuned
- **Language(s) (NLP):** English
- **Finetuned from model:** meta-llama/Llama-2-7b-hf

# Results
|       Epoch       | Accuracy on the testing set | Model Link |
|:---:|:----:|:----:|
| 1 | 0.609 | 🤗 [Hugging Face](https://huggingface.co/shuyuej/metamath_lora_llama2_7b) |
| 2 | 0.635 | 🤗 [Hugging Face](https://huggingface.co/shuyuej/metamath_lora_llama2_7b_2_epoch)  |
| 3 | 0.641 | 🤗 [Hugging Face](https://huggingface.co/shuyuej/metamath_lora_llama2_7b_3_epoch) |
| 4 | 0.641 | 🤗 [Hugging Face](https://huggingface.co/shuyuej/metamath_lora_llama2_7b_4_epoch) |

# Deployment
```python
# Load the Pre-trained LoRA Adapter
model.load_adapter("shutter/metamath_lora_llama2_7b_4_epoch")
model.enable_adapters()
```
