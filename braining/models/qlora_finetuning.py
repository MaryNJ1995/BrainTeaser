# -*- coding: utf-8 -*-
import bitsandbytes as bnb
import torch
import transformers
from datasets.utils.logging import set_verbosity_info
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
from trl import SFTTrainer

from braining.models.helper import print_trainable_parameters


class LanguageModelLoader:
    def __init__(self, model_name, mode, arg, train_data, eval_data):
        self.model_args = None
        self.model_name = model_name
        self.arg = arg
        self.mode = mode
        self.train_data = train_data
        self.eval_data = eval_data
        # self.accelerator = accelerator
        self.model, self.tokenizer = self.setup_model()

    @staticmethod
    def create_bnb_config():
        """
        Creates and returns a BitsAndBytesConfig for model quantization.

        Returns:
            BitsAndBytesConfig: Configuration for model quantization with the following parameters:
                - load_in_4bit: Whether to load the model in 4-bit mode.
                - bnb_4bit_use_double_quant: Whether to use double quantization for 4-bit mode.
                - bnb_4bit_quant_type: The type of 4-bit quantization, e.g., "nf4".
                - bnb_4bit_compute_dtype: The data type used for 4-bit computation, e.g., torch.bfloat16.

        Examples:
            bnb_config = create_bnb_config()
        """
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=torch.bfloat16, )
        return bnb_config

    def create_peft_config(self):
        """
        Create Parameter-Efficient Fine-Tuning (PEFT) configuration for the given model.

        Args:

        Returns:
            LoraConfig: Configuration for parameter-efficient fine-tuning using Lora with the following parameters:
                - r: Dimension of the updated matrices.
                - lora_alpha: Parameter for scaling.
                - target_modules: Names of the target modules to apply Lora to.
                - lora_dropout: Dropout probability for layers.
                - bias: Bias type, e.g., "none".
                - task_type: Task type, e.g., "CAUSAL_LM".

        Examples:
            model = AutoModelForCausalLM.from_pretrained("gpt2")
            lm_loader = LanguageModelLoader("gpt2")
            peft_config = lm_loader.create_peft_config(model)
        """
        config = LoraConfig(r=self.arg.lora_r,  # dimension of the updated matrices
                            lora_alpha=self.arg.lora_alpha,  # parameter for scaling
                            target_modules=self.find_all_linear_names(),  # target_modules=find_all_linear_names(model)
                            lora_dropout=self.arg.lora_dropout,  # dropout probability for layers
                            bias="none", task_type="CAUSAL_LM", )
        return config

    def peft_model_initializing(self):
        """
        Initialize a language model for Parameter-Efficient Fine-Tuning (PEFT).

        Args:

        Returns:
            Model: Parameter-Efficient Fine-Tuning initialized model.

        Examples:
            arg_namespace = Namespace(pad_token="[PAD]", eos_token="[EOS]", bos_token="[BOS]", unk_token="[UNK]")
            lm_loader = LanguageModelLoader("gpt2")
            peft_model = lm_loader.peft_model_initializing(arg_namespace)
        """
        peft_config = self.create_peft_config()
        model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=False)
        peft_model = get_peft_model(model, peft_config)
        return peft_model

    # def create_peft_model(self):
    # peft_model = AutoPeftModelForCausalLM.from_pretrained(self.model_name, low_cpu_mem_usage=True,
    #                                                       torch_dtype=torch.bfloat16, load_in_4bit=True, )
    # peft_model.config.use_cache = False
    # return peft_model

    # SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py
    def find_all_linear_names(self):
        """
        Find all linear module names in the given language model.

        Args:

        Returns:
            list: List of linear module names.

        Examples:
            model = AutoModelForCausalLM.from_pretrained("gpt2")
            lm_loader = LanguageModelLoader("gpt2")
            linear_module_names = lm_loader.find_all_linear_names(model)
        """
        cls = (bnb.nn.Linear4bit)  # if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
        lora_module_names = set()
        for name, module in self.model.named_modules():
            if isinstance(module, cls):
                names = name.split(".")
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        #    if 'lm_head' in lora_module_names:  # needed for 16-bit
        #        lora_module_names.remove('lm_head')
        return list(lora_module_names)

    def setup_model(self):
        """
        Loads a pre-trained language model and tokenizer with specified configurations.

        Args:
            arg (Namespace): An object containing special token values such as 'pad_token', 'eos_token',
                            'bos_token', and 'unk_token'.

        Returns:
            Tuple: A tuple containing the loaded language model (AutoModelForCausalLM) and tokenizer (AutoTokenizer).

        Raises:
            SomeException: Description of the exception, if any.

        Examples:
            arg = Namespace(pad_token="[PAD]", eos_token="[EOS]", bos_token="[BOS]", unk_token="[UNK]")
            model_name = "gpt2"
            bnb_config = QuantizationConfig(...)
            model, tokenizer = setup_model(arg, model_name, bnb_config)
        """
        # parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
        # model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        num_evaluate_steps = int(
            len(self.train_data) / (self.arg.per_device_train_batch_size * self.arg.gradient_accumulation_steps))
        print("%%%%num_evaluate_steps:%%%%", num_evaluate_steps)
        self.model_args2 = transformers.TrainingArguments(output_dir=self.arg.out_dir,
                                                          per_device_train_batch_size=self.arg.per_device_train_batch_size,
                                                          per_device_eval_batch_size=self.arg.per_device_train_batch_size,
                                                          evaluation_strategy=self.arg.evaluation_strategy,
                                                          load_best_model_at_end=True, logging_steps=num_evaluate_steps,
                                                          gradient_accumulation_steps=self.arg.gradient_accumulation_steps,
                                                          num_train_epochs=self.arg.num_train_epochs,
                                                          # report_to="tensorboard",
                                                          optim="paged_adamw_32bit",
                                                          learning_rate=self.arg.learning_rate, group_by_length=True,
                                                          fp16=True, ddp_find_unused_parameters=False, bf16=False,
                                                          max_grad_norm=self.arg.max_grad_norm,
                                                          warmup_ratio=self.arg.warmup_ratio, logging_first_step=True,
                                                          lr_scheduler_type="cosine", tf32=False,
                                                          metric_for_best_model="loss", save_total_limit=2,
                                                          save_steps=num_evaluate_steps, )
        self.model_args = transformers.TrainingArguments(per_device_train_batch_size=6, gradient_accumulation_steps=4,
                                                         learning_rate=5e-5, lr_scheduler_type="cosine", max_steps=200,
                                                         save_strategy="no", logging_steps=1,
                                                         output_dir=self.arg.out_dir, num_train_epochs=10,
                                                         optim="paged_adamw_32bit", warmup_steps=100, bf16=False,
                                                         save_total_limit=2)
        n_gpus = torch.cuda.device_count()

        # configuration = MistralConfig()
        model = AutoModelForCausalLM.from_pretrained(self.model_name,  # config=configuration,
                                                     quantization_config=self.create_bnb_config(), device_map="auto",
                                                     torch_dtype=torch.bfloat16,  # '':torch.cuda.current_device()}
                                                     load_in_8bit=True,
                                                     # dispatch efficiently the model on the available ressources
                                                     max_memory={i: f"{self.arg.max_memory}MB" for i in range(n_gpus)},
                                                     # Requires Flash Attention 2 installation
                                                     use_flash_attention_2=False, )  # .to(device)
        model.config.use_cache = False
        # ref_model = AutoModelForCausalLM.from_pretrained(self.model_name,  # config=configuration,
        #                                                  quantization_config=self.create_bnb_config(),
        #                                                  device_map="auto", torch_dtype=torch.bfloat16,
        #                                                  # '':torch.cuda.current_device()}
        #                                                  load_in_8bit=True,
        #                                                  # dispatch efficiently the model on the available ressources
        #                                                  max_memory={i: f"{self.arg.max_memory}MB" for i in
        #                                                              range(n_gpus)},
        #                                                  # Requires Flash Attention 2 installation
        #                                                  use_flash_attention_2=False, )  # .to(device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        # (, padding_side="left",
        # pad_to_multiple_of=arg.max_length,
        # model_max_length=arg.max_length,
        # add_eos_token=True, model_max_length=512, use_fast=False,
        # trust_remote_code=True, )  # .to(device)
        if self.mode == "train":
            if tokenizer.pad_token is None:
                self.smart_tokenizer_and_embedding_resize(special_tokens_dict=dict(pad_token="[PAD]"),
                                                          tokenizer=tokenizer, model=model, )

            special_tokens_dict = {}
            special_tokens = ["pad_token", "eos_token", "bos_token", "unk_token"]
            for token in special_tokens:
                if getattr(tokenizer, f"{token}") is None:
                    special_tokens_dict[token] = getattr(self.arg, token)
            if "llama" in self.model_name:
                tokenizer.add_special_tokens({"eos_token": self.arg.eos_token, "bos_token": self.arg.eos_token,
                                              "unk_token": self.arg.eos_token, })
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = tokenizer.pad_token_id
            tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

            if torch.cuda.device_count() > 1:
                model.is_parallelizable = True
                model.model_parallel = True
            return model, tokenizer
        else:
            return model, tokenizer  ### Response:  their same man.  ### End.

    @staticmethod
    def smart_tokenizer_and_embedding_resize(special_tokens_dict: {}, tokenizer: transformers.PreTrainedTokenizer,
                                             model: transformers.PreTrainedModel, ):
        """Resize tokenizer and embedding.

        Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
        """
        num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
        for param in model.parameters():
            param.requires_grad = False  # freeze the model - train adapters later
            if param.ndim == 1:
                # cast the small parameters (e.g. layernorm) to fp32 for stability
                param.data = param.data.to(torch.float32)

        model.gradient_checkpointing_enable()  # reduce number of stored activations
        model.enable_input_require_grads()
        if num_new_tokens > 0:
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

    def forward(self):
        # parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
        # model_args, _, _ = parser.parse_args_into_dataclasses()

        model, tokenizer = self.setup_model()

        # Create PEFT configuration
        peft_config = self.create_peft_config()
        # Activate nested quantization for 4-bit base models (double quantization)
        use_nested_quant = False
        # Check GPU compatibility with bfloat16
        bnb_4bit_compute_dtype = "float16"
        use_4bit = True
        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
        if compute_dtype == torch.float16 and use_4bit:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16: accelerate training with bf16=True")
                print("=" * 80)
        # Prepare model for kbit training
        # model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        # Check for multiple GPUs and set model properties accordingly
        model = self.peft_model_initializing()
        # Create PEFT model
        # model = get_peft_model(model, peft_config)
        print_trainable_parameters(model)
        # model = accelerator.prepare_model(model)
        # Create and initialize the trainer
        trainer = SFTTrainer(model=model, train_dataset=self.train_data, eval_dataset=self.eval_data,
                             peft_config=peft_config, max_seq_length=self.arg.max_length, tokenizer=tokenizer,
                             args=self.model_args, dataset_text_field="instruction", )
        # peft_model = self.create_peft_model()
        #
        # trainer = DPOTrainer(model, ref_model, args=self.model_args, train_dataset=self.train_data,
        #                          tokenizer=tokenizer, beta=0.1, max_prompt_length=1024,
        #                          max_length=self.arg.max_length )
        # https://towardsdatascience.com/fine-tune-a-mistral-7b-model-with-direct-preference-optimization-708042745aac
        # https://colab.research.google.com/drive/15iFBr1xWgztXvhrj5I9fBv20c7CFOPBE?usp=sharing#scrollTo=MCD77GZ60DOT
        # https://github.com/mzbac/llama2-fine-tune/blob/master/dpo_trainer.py
        # Disable caching for both PEFT model and original model
        model.config.use_cache = False

        # Set verbosity to info
        set_verbosity_info()

        # Train the model
        trainer.train()

        # Save the last checkpoint of the model
        print("Saving last checkpoint of the model...")
        trainer.save_model(
            self.arg.out_dir)  # trainer.save_state()  # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=self.arg.out_dir)  # Flush memory  # del dpo_trainer, model, ref_model  # torch.cuda.empty_cache()  #  # # Reload model in FP16 (instead of NF4)  # base_model = AutoModelForCausalLM.from_pretrained(  #     model_name,  #     return_dict=True,  #     torch_dtype=torch.float16,  # )  # tokenizer = AutoTokenizer.from_pretrained(model_name)  #  # # Merge base model with the adapter  # model = PeftModel.from_pretrained(base_model, "final_checkpoint")  # model = model.merge_and_unload()  #  # # Save model and tokenizer  # model.save_pretrained("new_model")  # tokenizer.save_pretrained("new_model")

    def run_and_merge(self, model_name):
        # Reload model in FP16 and merge it with LoRA weights
        base_model = AutoModelForCausalLM.from_pretrained(self.model, low_cpu_mem_usage=True, return_dict=True,
                                                          torch_dtype=torch.float16, device_map="auto", )
        model = PeftModel.from_pretrained(base_model, self.arg.out_dir)
        model = model.merge_and_unload()

        # Reload tokenizer to save it
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model.push_to_hub(self.arg.out_dir, use_temp_dir=False)
        tokenizer.push_to_hub(self.arg.out_dir, use_temp_dir=False)  #
