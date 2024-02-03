# -*- coding: utf-8 -*-
import logging
import os

import numpy as np
import plotly.graph_objects as go
import torch
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from datasets import Dataset
from peft import PeftModel
from torch.distributed.fsdp.fully_sharded_data_parallel import (FullOptimStateDictConfig, FullStateDictConfig, )
from tqdm import tqdm
from transformers import set_seed, AutoTokenizer

from braining import LanguageModelLoader, generate_prompt, dump_txt
from braining.config import BaseConfig
from braining.inference.inference import PostProcess
from braining.config.config import OpenAIDecodingArguments, ModelArguments
from braining.inference.helper import read_labels, calculate_accuracy

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False), )

accelerator = Accelerator(device_placement=False, mixed_precision="fp16", cpu=False, fsdp_plugin=fsdp_plugin)

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()
    set_seed(ARGS.seed)

    DEV_DATA = np.load(os.path.join(ARGS.data_path, ARGS.data_mode[0], ARGS.dev_data_path), allow_pickle=True)
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
    SOLAR = "/mnt/disk2/LanguageModels/SOLAR-10.7B-v1.0"
    MISTRAL = "/mnt/disk2/LanguageModels/Mistral-7B-v0.1"
    LAMA13 = "/mnt/disk2/LanguageModels/llama-2-13b"
    LM_LOADER = LanguageModelLoader(ModelArguments.model_name_or_path, MODE, ARGS, instructed_DEV_DATA,
                                    instructed_DEV_DATA)
    # --------------------------------------------- Run model -----------------------------------------------
    LM_LOADER.create_bnb_config()
    BASE_MODEL, TOKENIZER1 = LM_LOADER.setup_model()
    TOKENIZER = AutoTokenizer.from_pretrained(
        "/mnt/disk2/maryam.najafi/Project_LLMFineTune/testak/assets/saved_model2Mistral-T5-7B-v1/", add_bos_token=True,
        trust_remote_code=True)
    TOKENIZER.pad_token = TOKENIZER.eos_token
    TOKENIZER.pad_token_id = TOKENIZER.eos_token_id
    import transformers

    PEFT_MODEL = PeftModel.from_pretrained(BASE_MODEL,
                                           "/mnt/disk2/maryam.najafi/Project_LLMFineTune/testak/assets/saved_model2Mistral-T5-7B-v1/")
    # "/mnt/disk2/maryam.najafi/Project_LLMFineTune/testak/assets/saved_model2/")
    PEFT_MODEL.eval()
    FINAL_INDEX_LIST = list()
    parser = transformers.HfArgumentParser(OpenAIDecodingArguments)
    decoding_args = parser.parse_args_into_dataclasses()
    config = dict(temperature=0.2, top_k=40, top_p=0.95, do_sample=True, num_beams=1, repetition_penalty=1.2,
                  max_new_tokens=100)
    for index, sample in enumerate(tqdm(instructed_DEV_DATA)):
        print("\n1111111111111111111111111111\n", sample["instruction"])
        token_sample = TOKENIZER(sample["instruction"], return_tensors="pt").to("cuda")

        MODEL_OUTPUT = PEFT_MODEL.generate(**token_sample, **config, pad_token_id=TOKENIZER.eos_token_id, )
        # extracted_dict = extract_human_assistant(sample["instruction"])
        # print("extracted_dict", extracted_dict["human"])
        with torch.no_grad():
            results = [TOKENIZER.decode(gen, skip_special_tokens=True) for gen in MODEL_OUTPUT]
            print("\n123123123123123123123123\n", results)
            print("\n000000000000000000000000\n", sample["instruction"])
            # print("\n2222222222222222222222222222222222222\n", results[0])
            POSTPROCESSOR = PostProcess(results[0], sample["instruction"])
            MOST_SIMILAR_CHOICE, MOST_SIMILAR_INDEX = POSTPROCESSOR.process_generated_output()
            print("Most Similar Choice to the given generated:", MOST_SIMILAR_CHOICE)
            print("Most Similar Index to the given generated:", MOST_SIMILAR_INDEX)
            # print("************************", DEV_DATA[index]["choice_list"])
            print("\n\n\n\n\n")
            FINAL_INDEX_LIST.append(MOST_SIMILAR_INDEX)

    print(FINAL_INDEX_LIST)
    dump_txt('answer_sen.txt', FINAL_INDEX_LIST)

    # Read labels from files
    ground_truth_labels = read_labels('answer_sent.txt')
    predicted_labels = read_labels('answer_sen.txt')

    # Calculate accuracy
    accuracy = calculate_accuracy(ground_truth_labels, predicted_labels)

    print(f"Accuracy: {accuracy:.2f}%")

    # --------------------------------------------------------------------------------visualizing:

    input_text = "I have a dream"
    inputs = TOKENIZER.encode_plus(input_text, return_tensors="pt")

    # Compute the original loss
    outputs = PEFT_MODEL(**inputs, labels=inputs["input_ids"])
    original_loss = outputs.loss.item()

    # Define two random directions
    direction1 = [torch.randn_like(p) for p in PEFT_MODEL.parameters()]
    direction2 = [torch.randn_like(p) for p in PEFT_MODEL.parameters()]

    # Normalize vectors
    for p, d1, d2 in zip(PEFT_MODEL.parameters(), direction1, direction2):
        norm_p = torch.linalg.norm(p.flatten())
        d1.div_(torch.linalg.norm(d1.flatten())).mul_(norm_p)
        d2.div_(torch.linalg.norm(d2.flatten())).mul_(norm_p)

    # Define the range to explore
    x = np.linspace(-1, 1, 20)
    y = np.linspace(-1, 1, 20)
    X, Y = np.meshgrid(x, y)

    # Prepare to collect the losses
    Z = np.zeros_like(X)

    # Compute loss for each direction
    for i in tqdm(range(x.size), desc="x progress"):
        for j in tqdm(range(y.size), desc="y progress", leave=False):
            # Perturb the model parameters
            for p, d1, d2 in zip(PEFT_MODEL.parameters(), direction1, direction2):
                p.data.add_(x[i] * d1 + y[j] * d2)

            # Compute the loss
            outputs = PEFT_MODEL(**inputs, labels=inputs['input_ids'])
            Z[i, j] = outputs.loss.item()

            # Revert the model parameters
            for p, d1, d2 in zip(PEFT_MODEL.parameters(), direction1, direction2):
                p.data.sub_(x[i] * d1 + y[j] * d2)

    # Create 3D plot
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, showscale=False, )])
    fig.update_layout(title="GPT-2's Loss Landscape", autosize=True, width=1000, height=600,  # scene=dict(
                      #     xaxis=dict(visible=False),
                      #     yaxis=dict(visible=False),
                      #     zaxis=dict(visible=False),
                      # )
                      )
    fig.show()
