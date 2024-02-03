# -*- coding: utf-8 -*-
import logging

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from datasets import Dataset
from torch.distributed.fsdp.fully_sharded_data_parallel import (FullOptimStateDictConfig, FullStateDictConfig, )
from transformers import set_seed

from braining import (BaseConfig, LanguageModelLoader, load_npy, generate_prompt, )
from braining.config.config import ModelArguments

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()
    set_seed(ARGS.seed)

    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False), )

    accelerator = Accelerator(device_placement=False, mixed_precision="fp16", cpu=False, fsdp_plugin=fsdp_plugin, )
    # --------------------------------------------- Create Data ------------------------------------------------
    TRAIN_DATA = load_npy("/mnt/disk2/maryam.najafi/Project_LLMFineTune/testak/data/WPV2/final_train.npy")
    logging.warning("\n Train Data length is: {}".format(len(TRAIN_DATA)))
    logging.warning("\n Train Data sample: {}".format(TRAIN_DATA))
    DEV_DATA = load_npy("/mnt/disk2/maryam.najafi/Project_LLMFineTune/testak/data/WPV2/final_dev.npy")
    logging.warning("\n Dev Data length is: {}".format(len(DEV_DATA)))


    instructed_TRAIN_DATA = generate_prompt(TRAIN_DATA)
    instructed_TRAIN_DATA = Dataset.from_list(instructed_TRAIN_DATA)
    logging.warning("\n Train prompted Data length is: {}".format(len(instructed_TRAIN_DATA)))
    logging.warning("\n  Tain prompted Data sample is: {}".format((instructed_TRAIN_DATA[13])))

    instructed_DEV_DATA = generate_prompt(DEV_DATA)
    instructed_DEV_DATA = Dataset.from_list(instructed_DEV_DATA)
    logging.warning("\n Dev prompted Data length is: {}".format(len(instructed_DEV_DATA)))
    logging.warning("\n  Dev prompted Data sample is: {}".format((instructed_DEV_DATA[13])))

    # --------------------------------------------- Load model -----------------------------------------------
    # Create an instance of LanguageModelLoader
    mode = "train"
    lm_loader = LanguageModelLoader(ModelArguments.model_name_or_path, mode, ARGS)
    # --------------------------------------------- Run model -----------------------------------------------
    lm_loader.forward(instructed_TRAIN_DATA, instructed_DEV_DATA)
    print("Train Is Completed!")
