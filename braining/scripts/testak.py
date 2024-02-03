import re

import torch
from datasets import Dataset
from peft import PeftModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from braining import BaseConfig, load_npy, generate_prompt
from config.config import ModelArguments
from tqdm import tqdm


def remove_text_a(text_a, text_b):
    return text_b.replace(text_a, '')


def extract_answer(answer: str):
    # response_pattern = re.compile(r'### Response:(.*?)(?=### Question:|$)', re.DOTALL)
    # response_pattern = re.compile(r'### Response:\n(.+)\n', re.DOTALL)
    # response_pattern = re.compile(r'### Response:\n(.*?)', re.DOTALL)
    response_pattern = re.compile(r'### Response:(.*?)###', re.DOTALL)
    ### Response:(.*?)###
    match = response_pattern.search(answer)
    if match:
        extracted_response = match.group(1).strip()
        return extracted_response
    else:
        print("in function extract_answer")
        print(answer)
        return None


def find_most_similar(response, choices):
    vectorizer = TfidfVectorizer()
    # Combine response and choices for vectorization
    all_text = [response] + choices
    vectors = vectorizer.fit_transform(all_text)
    # Calculate cosine similarity between the response and each choice
    similarities = cosine_similarity(vectors[0], vectors[1:]).flatten()
    # Find the index of the most similar choice
    most_similar_index = similarities.argmax()
    # Return the most similar choice
    return most_similar_index


if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()

    DEV_DATA = load_npy(ARGS.dev_data_path)
    TRAIN_DATA = load_npy(ARGS.train_data_path)
    PROCESSED_DEV_DATA = generate_prompt(main_samples=DEV_DATA, mode="test")[:3]
    DEV_DATASET = Dataset.from_list(PROCESSED_DEV_DATA)

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16)
    base_model = AutoModelForCausalLM.from_pretrained(ModelArguments.model_name_or_path,  # Llama 2 7B, same as before
        quantization_config=bnb_config,  # Same quantization config as before
        device_map="auto", trust_remote_code=True, token=True)

    tokenizer = AutoTokenizer.from_pretrained(ModelArguments.model_name_or_path, add_bos_token=True,
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id


    peft_model = PeftModel.from_pretrained(base_model, "/mnt/disk2/maryam.najafi/Project_LLMFineTune/testak/assets/saved_model2")
    peft_model.eval()

    RESULTS = []
    for index, sample in enumerate(tqdm(DEV_DATASET)):
        print(index / len(DEV_DATASET) * 100)
        tokenized_text = tokenizer(sample["instruction"], return_tensors="pt").to("cuda")
        output = peft_model.generate(**tokenized_text, max_new_tokens=100,
                                     pad_token_id=tokenizer.eos_token_id)

        with torch.no_grad():
            results = [tokenizer.decode(res, skip_special_tokens=True) for res in output]
            print("dfjvkjfvnkjdfvndfjvdfnvkdfnvkdfjnvkdfjvn", results)
            print("dfjvkjfvnkjdfvndfjvdfnvkdfnvkdfjnvkdfjvn&&&&&&&", results[0])
            generated_text = remove_text_a(text_a=sample["instruction"], text_b=results[0])
            print("5555555555555555555555555555555555555555555555555555555555",generated_text)

            generated_text = "### Response:" + generated_text
            print("66666666666666666666666666666666666666666666666666666666666",generated_text)

            print(generated_text)
            extracted_answer = extract_answer(generated_text)  # .strip().replace("'", "")
            print("7777777777777777777777777777777777777777777777777777777777",generated_text)

            if extracted_answer in DEV_DATA[index]["choice_list"]:
                RESULTS.append(DEV_DATA[index]["choice_list"].index(extracted_answer))

            else:
                founded_answer = find_most_similar(extracted_answer, DEV_DATA[index]["choice_list"])
                RESULTS.append(founded_answer)
                print(extracted_answer)
                print(DEV_DATA[index]["choice_list"])
                print("founded_answer")
                print(DEV_DATA[index]["choice_list"][int(founded_answer)])
    print(RESULTS[:3])
    print("djc;djc;ldjs;clkjdskcjd;lscjdkl")
    # Open a file in write mode ('w')
    with open('answer_sen.txt', 'w') as file:
        # Iterate through the list and write each element on a new line
        for item in RESULTS:
            file.write(f"{item}\n")
