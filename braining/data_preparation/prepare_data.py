# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split

from braining.data_loader.read_files import write_json, load_npy


def create_prompt_instruction(sample):
    return f"""### Instruction:
   Use the input below to create an instruction, which could have been used to generate the input using an LLM.

   ### Input
   {sample['response']}

   ### Response:
   {sample['instruction']}
   """


def extract_human_assistant(text):
    human_start = "### Human:"
    assistant_start = "### Assistant:"

    human_index = text.find(human_start)
    assistant_index = text.find(assistant_start)

    if human_index != -1 and assistant_index != -1:
        human_text = text[human_index + len(human_start): assistant_index].strip()
        assistant_text = text[assistant_index + len(assistant_start):].strip()

        return {"human": human_text, "assistant": assistant_text}

    return None


def redesigned_aplaca_data(original_data):
    data_pack = []

    for entry in original_data:
        # Assuming the structure of each entry is {"text": "###human: ... ###assistance: ..."}
        transformed_entry = extract_human_assistant(entry["text"])
        # Extract values for "human" and "assistance"
        data_pack.append(transformed_entry)

    return data_pack


def prompt_single_phase_creator(qa_dataset):
    """Given a sample dictionary with key "conversations", format the conversation into a prompt.
    Args:
      qa_dataset: a Persian QA dataset
    Returns:
      sample: sample dictionary with "text" key for the formatted prompt.
    """
    INTRO = "Below is a conversation between a user and you. "
    INSTRUCTION = ("### Instruction: You are an excellent question responder. "
                   "Generate a related response to the user's query based on the document. ")
    QUERY = "Input Question: "
    END_KEY = " ### End"
    final_sample = []
    for conversation in qa_dataset:
        intro = f"{INTRO}"
        instruction = f"{INSTRUCTION}"
        input_query = f"{QUERY}\n{conversation['human']}"
        response_key = "### Answer: "
        conversation = f"{response_key}\n{conversation['assistant']}"
        end = f"{END_KEY}"
        sections = [intro, instruction, input_query, conversation, end]
        formatted_sample = "\n".join(sections)
        final_sample.append({"instruction": formatted_sample})
    return final_sample


def split_json_dataset(input_dataset, train_data_dir, test_data_dir):
    transformed_data = redesigned_aplaca_data(input_dataset)

    # Extract the "human" and "assistant" data into separate lists
    human_data = [item["human"] for item in transformed_data]
    assistant_data = [item["assistant"] for item in transformed_data]

    # Perform the train-test split
    train_human, test_human, train_assistant, test_assistant = train_test_split(human_data, assistant_data,
                                                                                test_size=0.2, random_state=42)

    # Combine the split data back into dictionaries if needed
    train_data = [{"human": h, "assistant": a} for h, a in zip(train_human, train_assistant)]
    test_data = [{"human": h, "assistant": a} for h, a in zip(test_human, test_assistant)]
    write_json(train_data_dir, train_data)
    write_json(test_data_dir, test_data)


def create_instruction(sample, few_shot_samples, num_shots, mode):
    QUERY_KEY = "### Input Question: "
    CHOICE_KEY = "### ChoiceList: "
    RESPONSE_KEY = "### Response: "
    END_KEY = " ### End."

    query = f"{QUERY_KEY} \n {sample['question']}"
    choice_list = f"{CHOICE_KEY} \n {sample['choice_list']}"
    INSTRUCTION = ("You are a contextual multiple-choice question-answering chatbot. "
                   f"Choose the most suitable answer for the following question from the choice_list{ choice_list}. "
                   "Be cautious, as the question's context may pose a distraction. Think deep and step by step before finalizing your choice. "
                   "Your answer should come after '### Response: \n' and should be same as one of choice_list")
    #     "### Instruction: As an professional multiple-choice question-answering system, your task is to meticulously select the most fitting response from the provided choice_list for the upcoming question. "
    #     " Be cautious, as the question's context may pose a distraction. Think deeply before finalizing your choice. "
    #     # "- Pay equal attention to both correct and incorrect options to enhance your selection skills. "
    #     # "- Please generate tokens ONLY in English "
    #     # "- Please do NOT generate duplicate tokens at all. "
    #     "- Keep the answer short and concise. "
    #     "- please be sure that your answer is in the choice_list"
    #     "- Please generate ONE answer only after ### Response: "
    # )
    # if few_shot_samples and mode == "train":
    #     FEW_INS = "Consider the following samples as few shot samples: \n"
    #     END_INS = "these sample are the few-shots and now the main question: \n"
    #     # Add random examples
    #     few_shot_instructions = [
    #         f" {QUERY_KEY} \n {sample['question']} \n {CHOICE_KEY} \n {sample['choice_list']} \n {RESPONSE_KEY} \n {sample['answer']}"
    #         for _ in range(num_shots) if (sample := random.choice(few_shot_samples))]
    #     INSTRUCTION = "\n".join([INSTRUCTION, FEW_INS, *few_shot_instructions, END_INS])
    if not isinstance(INSTRUCTION, str):
        raise ValueError(f"prompt should be an str but got {type(INSTRUCTION)}")

    if mode == "train":
        response = f"{RESPONSE_KEY} \n {sample['answer']}"
        final_prompt = (" ".join([INSTRUCTION, query, choice_list, response, END_KEY]))
        return {"instruction": final_prompt}

    elif mode == "test":
        final_prompt = (" ".join([INSTRUCTION, query, choice_list, RESPONSE_KEY]))
        return {"instruction": final_prompt}


def create_DPO_instruction(sample, mode):
    QUERY_KEY = "### Input Question: "
    CHOICE_KEY = "### ChoiceList: "
    RESPONSE_KEY = "### Response: "
    START_KEY = "<|im_start|>"
    END_KEY = "<|im_end|>"
    query = f"{START_KEY} user\n {sample['question']}, {END_KEY}"
    choice_list = f"{CHOICE_KEY} \n {sample['choice_list']}"
    INSTRUCTION = ("You are a contextual multiple-choice question-answering chatbot. "
                   f"Choose the most suitable answer for the following question from the choice_list{ choice_list}. "
                   "Be cautious, as the question's context may pose a distraction. Think deep and step by step before finalizing your choice. "
                   "Your answer should come after '### Response: \n' and should be same as one of choice_list")
    #     "### Instruction: As an professional multiple-choice question-answering system, your task is to meticulously select the most fitting response from the provided choice_list for the upcoming question. "
    #     " Be cautious, as the question's context may pose a distraction. Think deeply before finalizing your choice. "
    #     # "- Pay equal attention to both correct and incorrect options to enhance your selection skills. "
    #     # "- Please generate tokens ONLY in English "
    #     # "- Please do NOT generate duplicate tokens at all. "
    #     "- Keep the answer short and concise. "
    #     "- please be sure that your answer is in the choice_list"
    #     "- Please generate ONE answer only after ### Response: "
    #

    chosen = sample['answer'] + "<|im_end|>\n"
    rejected = str(sample['distractor1']) + "<|im_end|>\n"
    if mode == "train":
        response = f"{RESPONSE_KEY} \n {sample['answer']}"
        final_prompt = (" ".join([START_KEY, INSTRUCTION, END_KEY, query, START_KEY, "assistant\n", response, END_KEY]))
        return {"prompt": final_prompt, "chosen": chosen, "rejected": rejected}

    elif mode == "test":
        final_prompt = (" ".join([START_KEY, INSTRUCTION, END_KEY, query, START_KEY, "assistant\n", ]))
        return {"prompt": final_prompt, "chosen": chosen, "rejected": rejected}


def generate_prompt(main_samples, few_shot_samples=None, num_shots=None, mode="train"):
    instructions = []
    for data_sample in main_samples:
        # Generate the prompt using the separate function
        prompt = create_instruction(dict(data_sample), few_shot_samples, num_shots, mode)
        instructions.append(prompt)

    return instructions


# class SupervisedDataset(Dataset):
#     """Dataset for supervised fine-tuning."""
#
#     def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
#         super(SupervisedDataset, self).__init__()
#         logging.warning("Loading data...")
#         list_data_dict = jload(data_path)
#
#         logging.warning("Formatting inputs...")
#         prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
#         sources = [
#             prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
#             for example in list_data_dict]
#         targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]
#
#         logging.warning("Tokenizing inputs... This may take some time...")
#         data_dict = preprocess(sources, targets, tokenizer)
#
#         self.input_ids = data_dict["input_ids"]
#         self.labels = data_dict["labels"]
#
#     def __len__(self):
#         return len(self.input_ids)
#
#     def __getitem__(self, i) -> Dict[str, torch.Tensor]:
#         return dict(input_ids=self.input_ids[i], labels=self.labels[i])


if __name__ == '__main__':
    PROMPT_DICT = {"prompt_response": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"),
        "prompt_no_response": ("Below is an instruction that describes a task. "
                               "Write a response that appropriately completes the request.\n\n"
                               "### Instruction:\n{instruction}\n\n### Response:"), }

    data = load_npy("/mnt/disk2/maryam.najafi/testak/Braining/data/SPV2-train.npy")
    few_shot_sample = [{'id': 'SPV2-1',
                        'question': 'A woman shoots her husband. Then she holds him underwater for over 5 minutes. Finally, she hangs him. But 5 minutes later, they both go out and enjoy a wonderful dinner together. How can this be?',
                        'answer': 'The woman was a photographer. She shot a picture of her husband, developed it, and hung it up to dry.',
                        'distractor1': 'The woman gets arrested for murder after dinner.',
                        'distractor2': 'The woman gets a new partner.', 'distractor(unsure)': 'None of above.',
                        'label': 2, 'choice_list': ['The woman gets arrested for murder after dinner.',
                                                    'The woman gets a new partner.',
                                                    'The woman was a photographer. She shot a picture of her husband, developed it, and hung it up to dry.',
                                                    'None of above.'], 'choice_order': [1, 2, 0, 3]},

                       {'id': 'SPV2-25',
                        'question': 'Brad started through the dirty sort-shared window on the 22nd floor of the office tower. Overcome with depression he slid the window open and jumped through it. It was a sheer drop outside the building to the ground. Miraculously after he landed he was completely uninjured. Since there was nothing to cushion his fall or slow his descent, how could he have survived the fall?',
                        'answer': 'Brad was so sick and tired of window washing, he opened the window and jumped inside.',
                        'distractor1': 'The ground outside the building is wet.',
                        'distractor2': 'Consistent exercise has made him a very strong man.',
                        'distractor(unsure)': 'None of above.', 'label': 1,
                        'choice_list': ['The ground outside the building is wet.',
                                        'Brad was so sick and tired of window washing, he opened the window and jumped inside.',
                                        'Consistent exercise has made him a very strong man.', 'None of above.'],
                        'choice_order': [1, 0, 2, 3]}]
    instruction = generate_prompt(data, few_shot_sample, num_shots=2, mode="test")
    print(instruction[0])
    print("\n")
    print(instruction[1])

    # RAW_DATA = load_jsonl(  # "/mnt/disk2/maryam.najafi/Project_LLMFineTune/data/data.jsonl"  # )  # train_data_path = "../data/train_data.json"  # test_data_path = "../data/test_data.json"  # split_json_dataset(RAW_DATA, train_data_path, test_data_path)
