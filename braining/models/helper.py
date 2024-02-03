# -*- coding: utf-8 -*-
import torch
import transformers
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BitsAndBytesConfig


def create_quantization_config():
    """

    :return:
    """
    bnb_config = BitsAndBytesConfig(load_in_8bit=True,  # load model in 8-bit precision
                                    load_in_4bit=True,  # load model in 4-bit precision
                                    bnb_4bit_quant_type="nf4",
                                    # pre-trained model should be quantized in 4-bit NF format
                                    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16, )
    return bnb_config


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} ||"
          f" all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


def tokenize_fn(tokenizer, example):
    context_length = tokenizer.model_max_length
    outputs = tokenizer(tokenizer.eos_token.join(example["text"]), truncation=False, return_tensors="pt",
                        pad_to_multiple_of=context_length, padding=True, )
    return {"input_ids": outputs["input_ids"].view(-1, context_length)}


def tokenize_prompt(prompt, tokenizer):
    """

    :param prompt:
    :param tokenizer:
    :return:
    """
    result = tokenizer(prompt, truncation=True, max_length=512, padding="max_length", )
    result["labels"] = result["input_ids"].copy()
    return result


def generate_and_tokenize_prompt(data_point, tokenizer):
    """

    :param data_point:
    :param tokenizer
    :return:
    """
    full_prompt = f"""Given a target sentence construct the underlying meaning representation of the input sentence
     as a single function with attributes and attribute values.
     This function should describe the target string accurately and the function must be one of the following
     ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute',
     'suggest', 'request_explanation', 'recommend', 'request_attribute'].
     The attributes must be one of the following: ['name', 'exp_release_date',
     'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective',
     'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']

    ### Target sentence:
    {data_point["target"]}

    ### Meaning representation:
    {data_point["meaning_representation"]}
    """
    return tokenize_prompt(full_prompt, tokenizer)


def extract_question(instruction_text):
    """
    
    Args:
        instruction_text:

    Returns:

    """
    start_marker = "\nInput Question: \n"
    end_marker = "\n ### End"
    start_index = instruction_text.find(start_marker) + len(start_marker)
    end_index = instruction_text.find(end_marker, start_index)

    if start_index != -1 and end_index != -1:
        extracted_text = instruction_text[start_index:end_index]
        return extracted_text
    else:
        return None


def retrieve_top_choice(self, generated_sample, choices, top_k):
    """
    This method retrieval top chunk.
    Args:
        query: The query received from the user.
        chunks: chunk_document function's output
        top_k: The number of top chunks that must be returned

    Returns: retrieve_top_chunk

    """
    # Encode query
    query_embedding = self.retriever_model.encode([generated_sample])

    # Encode chunks
    chunk_embeddings = self.retriever_model.encode(choices)

    # Calculate cosine similarities
    similarities = cosine_similarity(query_embedding, chunk_embeddings)

    # Retrieve top-K chunks
    top_choice_indices = similarities.argsort(axis=1)[:, -top_k:][0]
    top_chunk_documents = [choices[i] for i in top_choice_indices]

    return top_chunk_documents
