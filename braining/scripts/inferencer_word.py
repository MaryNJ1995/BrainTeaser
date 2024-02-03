# -*- coding: utf-8 -*-
import logging

import torch
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from datasets import Dataset
from peft import PeftModel
from torch.distributed.fsdp.fully_sharded_data_parallel import (FullOptimStateDictConfig, FullStateDictConfig, )
from tqdm import tqdm
from transformers import set_seed

from braining import LanguageModelLoader, load_npy, generate_prompt, dump_txt
from braining.config import BaseConfig
from braining.config.config import ModelArguments
from braining.inference.inference import PostProcess

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False), )

accelerator = Accelerator(device_placement=False, mixed_precision="fp16", cpu=False, fsdp_plugin=fsdp_plugin)

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()
    set_seed(ARGS.seed)
    DEV_DATA = load_npy("/mnt/disk2/maryam.najafi/Project_LLMFineTune/testak/data/WPV2/final_dev.npy")
    # "/mnt/disk2/maryam.najafi/testak/Braining/data/SP_eval_data_for_practice.npy")
    logging.warning("creating the formatted data")
    logging.warning("\n Dev Data length is: {}".format(len(DEV_DATA)))
    logging.warning("\n Train Data sample: {}".format(DEV_DATA))

    instructed_DEV_DATA = generate_prompt(DEV_DATA, mode="test")
    instructed_DEV_DATA = Dataset.from_list(instructed_DEV_DATA)
    logging.warning("\n Dev prompted Data length is: {}".format(len(instructed_DEV_DATA)))
    logging.warning("\n  Dev prompted Data sample is: {}".format((instructed_DEV_DATA[0])))
    logging.warning(f"{len(DEV_DATA)} is dev data len...")
    # --------------------------------------------- Create prompt -----------------------------------------------
    # retrieve_top_choice()
    MODE = "test"
    LM_LOADER = LanguageModelLoader(ModelArguments.model_name_or_path, MODE, ARGS, instructed_DEV_DATA,
                                    instructed_DEV_DATA)
    # --------------------------------------------- Run model -----------------------------------------------
    LM_LOADER.create_bnb_config()
    BASE_MODEL, TOKENIZER = LM_LOADER.setup_model()
    TOKENIZER.pad_token = TOKENIZER.eos_token
    TOKENIZER.pad_token_id = TOKENIZER.eos_token_id

    PEFT_MODEL = PeftModel.from_pretrained(BASE_MODEL,
                                           "/mnt/disk2/maryam.najafi/Project_LLMFineTune/testak/assets/saved_model")
    # "/mnt/disk2/maryam.najafi/Project_LLMFineTune/testak/assets/saved_model2/")
    PEFT_MODEL.eval()
    FINAL_INDEX_LIST = list()
    for index, sample in enumerate(tqdm(instructed_DEV_DATA)):
        token_sample = TOKENIZER(sample["instruction"], return_tensors="pt").to("cuda")

        MODEL_OUTPUT = PEFT_MODEL.generate(**token_sample, max_new_tokens=100, pad_token_id=TOKENIZER.eos_token_id, )
        # extracted_dict = extract_human_assistant(sample["instruction"])
        # print("extracted_dict", extracted_dict["human"])
        with torch.no_grad():
            results = [TOKENIZER.decode(gen, skip_special_tokens=True) for gen in MODEL_OUTPUT]
            POSTPROCESSOR = PostProcess(results[0], sample["instruction"])
            MOST_SIMILAR_CHOICE, MOST_SIMILAR_INDEX = POSTPROCESSOR.process_generated_output()
            print("Most Similar Choice to the given generated:", MOST_SIMILAR_CHOICE)
            print("Most Similar Index to the given generated:", MOST_SIMILAR_INDEX)
            print("************************", DEV_DATA[index]["choice_list"])
            print("\n\n\n\n\nENDEDNEDNEDNEDNEDNEDNEDNENDNENENENDND")
            FINAL_INDEX_LIST.append(MOST_SIMILAR_INDEX)

    print(FINAL_INDEX_LIST)
    dump_txt('answer_word.txt', FINAL_INDEX_LIST)
