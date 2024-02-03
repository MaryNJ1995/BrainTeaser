# TheBloke/llava-v1.5-13B-GPTQ

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "HuggingFaceH4/zephyr-7b-beta"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./")
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./")
# TheBloke/Mixtral-8x7B-v0.1-GPTQ
# mistralai/Mixtral-8x7B-v0.1
tokenizer.save_pretrained("/mnt/disk2/LanguageModels/zephyr-7b-beta", model_name)
model.save_pretrained("/mnt/disk2/LanguageModels/zephyr-7b-beta")  # TomGrc/FusionNet_7Bx2_MoE_14B
# fblgit/UNA-SOLAR-10.7B-Instruct-v1.0
# HuggingFaceH4/zephyr-7b-beta
