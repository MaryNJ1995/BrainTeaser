# -*- coding: utf-8 -*-
# ============================ Third Party libs ============================
import argparse
# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import torch
import transformers


class BaseConfig:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--model_name", type=str, default="Mistral-7B-v0.1")
        self.parser.add_argument("--base_model_path", type=str, default="mnt/disk2/LanguageModels/Mistral-7B-v0.1")
        # Mistral-7B-v0.1
        self.parser.add_argument("--pad_token", type=str, default="[PAD]")
        self.parser.add_argument("--eos_token", type=str, default="</s>")
        self.parser.add_argument("--bos_token", type=str, default="</s>")
        self.parser.add_argument("--unk_token", type=str, default="<unk>")

        self.parser.add_argument("--load_in_8bit", type=bool, default=False)
        self.parser.add_argument("--load_in_4bit", type=bool, default=True)
        self.parser.add_argument("--bnb_4bit_use_double_quant", type=bool, default=True)
        self.parser.add_argument("--num_train_epochs", type=int, default=15)
        self.parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
        self.parser.add_argument("--per_device_train_batch_size", type=int, default=4)
        self.parser.add_argument("--logging_steps", type=int, default=1)
        self.parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
        self.parser.add_argument("--max_length", type=int, default=1024)
        self.parser.add_argument("--lora_r", type=int, default=32)
        self.parser.add_argument("--lora_alpha", type=int, default=64)
        self.parser.add_argument("--lora_dropout", type=float, default=0.05)
        self.parser.add_argument("--seed", type=int, default=5318008)
        self.parser.add_argument("--max_memory", type=int, default=32000)
        self.parser.add_argument("--lora_rank", type=int, default=8)
        self.parser.add_argument("--optim", type=str, help="activates the paging for better memory management",
                                 default="paged_adamw_32bit", )
        self.parser.add_argument("--save_strategy", type=str, help="checkpoint save strategy", default="steps", )
        self.parser.add_argument("--evaluation_strategy", type=str, default="steps")

        self.parser.add_argument("--save_steps", type=int, help="checkpoint saves", default=50)
        self.parser.add_argument("--learning_rate", type=float, help="learning rate for AdamW optimizer",
                                 default=0.0002, )
        self.parser.add_argument("--max_grad_norm", type=float, help="maximum gradient norm (for gradient clipping)",
                                 default=0.3, )
        self.parser.add_argument("--max_steps", type=int, help="training will happen for 'max_steps' steps",
                                 default=500, )
        self.parser.add_argument("--model_max_length", type=int, help="max lenght used for model", default=512, )
        self.parser.add_argument("--warmup_ratio", type=float,
                                 help="steps for linear warmup from 0 " "to learning_rate", default=0.03, )
        self.parser.add_argument("--lr_scheduler_type", type=str, default="cosine")

    def add_path(self) -> None:
        self.parser.add_argument("--raw_data_path", type=str, default=Path(__file__).parents[2].__str__() + "/data/Raw")
        self.parser.add_argument("--data_path", type=str, default=Path(__file__).parents[2].__str__() + "/data/")
        self.parser.add_argument("--processed_data_path", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/data/processed_data.json", )
        self.parser.add_argument("--train_data_path", type=str, default="final_train.npy", )
        self.parser.add_argument("--dev_data_path", type=str, default="final_dev.npy")
        self.parser.add_argument("--data_mode", type=list, default=["WPV2", "SPV2", "All"], )
        self.parser.add_argument("--out_dir", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/assets/saved_model2/", )
        self.parser.add_argument("--out_dir2", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/assets/saved_model/", )

    def get_config(self):
        self.add_path()
        return self.parser.parse_args()


@dataclass
class ModelArguments:
    SOLAR: str = "/mnt/disk2/LanguageModels/SOLAR-10.7B-v1.0"
    MISTRAL: str = "/mnt/disk2/LanguageModels/Mistral-7B-v0.1"
    LAMA13: str = "/mnt/disk2/LanguageModels/llama-2-13b"
    model_name_or_path: Optional[str] = field(default="/mnt/disk2/LanguageModels/Mistral-T5-7B-v1")
    tokenizer_name_or_path: Optional[str] = field(default="/mnt/disk2/LanguageModels/Mistral-T5-7B-v1")


@dataclass
class DataArguments:
    data_path: str = field(default="/mnt/disk2/maryam.najafi/Project_LLMFineTune/data/processed_data.json",
                           metadata={"help": "Path to the training data."}, )


@dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 1800
    temperature: float = 0.2
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    suffix: Optional[str] = None
    logprobs: Optional[int] = None
    echo: bool = False


@dataclass
class TrainingArguments:
    output_dir: Optional[str] = field(default="")
    # per_device_train_batch_size:  Optional[int] = field(default=4)
    gradient_accumulation_steps: Optional[int] = field(default=2)
    num_train_epochs: Optional[int] = field(default=5)
    save_total_limit: Optional[int] = field(default=2)
    learning_rate: Optional[float] = field(default=2e-4)
    warmup_ratio: Optional[float] = field(default=0.03)
    max_grad_norm: Optional[float] = field(default=0.3)
    optim: Optional[str] = field(default="paged_adamw_32bit")
    lr_scheduler_type: Optional[str] = field(default="cosine")
    evaluation_strategy: Optional[str] = field(default="steps")
    save_strategy: Optional[str] = field(default="steps")
    metric_for_best_model: Optional[str] = field(default="loss")
    bf16: Optional[bool] = field(default=False)
    tf32: Optional[bool] = field(default=False)
    packing: Optional[bool] = field(default=False)
    group_by_length: Optional[bool] = field(default=True)
    load_best_model_at_end: Optional[bool] = field(default=True)
    logging_first_step: Optional[bool] = field(default=True)
    cache_dir: Optional[str] = field(default="")
    # optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(model_args)
