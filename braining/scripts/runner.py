# -*- coding: utf-8 -*-
import logging
import os

import numpy as np
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from datasets import Dataset
from torch.distributed.fsdp.fully_sharded_data_parallel import (FullOptimStateDictConfig, FullStateDictConfig, )
from transformers import set_seed

from braining import (BaseConfig, LanguageModelLoader, generate_prompt, )
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
    TRAIN_DATA = np.load(os.path.join(ARGS.data_path, ARGS.data_mode[0], ARGS.train_data_path), allow_pickle=True)
    logging.warning("\n Train Data length is: {}".format(len(TRAIN_DATA)))
    logging.warning("\n Train Data sample: {}".format(TRAIN_DATA))
    DEV_DATA = np.load(os.path.join(ARGS.data_path, ARGS.data_mode[0], ARGS.dev_data_path), allow_pickle=True)
    logging.warning("\n Dev Data length is: {}".format(len(DEV_DATA)))

    instructed_TRAIN_DATA = generate_prompt(main_samples=TRAIN_DATA, few_shot_samples=TRAIN_DATA, num_shots=2,
                                            mode="train")
    instructed_TRAIN_DATA = Dataset.from_list(instructed_TRAIN_DATA)
    instructed_TRAIN_DATA = instructed_TRAIN_DATA.map(batched=True)
    logging.warning("\n Train prompted Data length is: {}".format(len(instructed_TRAIN_DATA)))
    logging.warning("\n  Tain prompted Data sample is: {}".format((instructed_TRAIN_DATA[13])))

    instructed_DEV_DATA = generate_prompt(main_samples=DEV_DATA, few_shot_samples=DEV_DATA, num_shots=2, mode="train")
    instructed_DEV_DATA = Dataset.from_list(instructed_DEV_DATA)
    logging.warning("\n Dev prompted Data length is: {}".format(len(instructed_DEV_DATA)))
    logging.warning("\n  Dev prompted Data sample is: {}".format((instructed_DEV_DATA[13])))

    # --------------------------------------------- Load model -----------------------------------------------
    # Create an instance of LanguageModelLoader
    mode = "train"
    print("model_name_or_path", ModelArguments.model_name_or_path)
    lm_loader = LanguageModelLoader(ModelArguments.model_name_or_path, mode, ARGS, instructed_TRAIN_DATA,
                                    instructed_DEV_DATA)
    # --------------------------------------------- Run model -----------------------------------------------
    lm_loader.forward()
    print("Train Is Completed!")
