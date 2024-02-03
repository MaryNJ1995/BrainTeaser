from typing import List, Optional
from cog import BasePredictor, Input #https://github.com/kurtseifried/cog_stanford_alpaca/blob/main/predict.py
#https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-Tune-an-LLM-Part-1-Preparing-a-Dataset-for-Instruction-Tuning--Vmlldzo1NTcxNzE2
from transformers import LLaMAForCausalLM, LLaMATokenizer
import torch

# from train import PROMPT_DICT

CACHE_DIR = 'alpaca_out'
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
PROMPT = PROMPT_DICT['prompt_no_input']

class Predictor(BasePredictor):
    def setup(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = LLaMAForCausalLM.from_pretrained("alpaca_out", cache_dir=CACHE_DIR, local_files_only=True)
        self.model = self.model
        self.model.to(self.device)
        self.tokenizer = LLaMATokenizer.from_pretrained("alpaca_out", cache_dir=CACHE_DIR, local_files_only=True)

    def predict(
        self,
        prompt: str = Input(description=f"Prompt to send to LLaMA."),
        n: int = Input(description="Number of output sequences to generate", default=1, ge=1, le=5),
        total_tokens: int = Input(
            description="Maximum number of tokens for input + generation. A word is generally 2-3 tokens",
            ge=1,
            default=2000
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
            ge=0.7,
            le=5,
            default=0.75,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.01,
            le=1.0,
            default=1.0
        ),
        repetition_penalty: float = Input(
            description="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it.",
            ge=0.01,
            le=5,
            default=1
        )
        ) -> List[str]:
        format_prompt = PROMPT.format_map({'instruction': prompt})
        input = self.tokenizer(format_prompt, return_tensors="pt").input_ids.to(self.device)

        outputs = self.model.generate(
            input,
            num_return_sequences=n,
            max_length=total_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
        out = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # removing prompt b/c it's returned with every input
        out = [val.split('Response:')[1] for val in out]
        return out