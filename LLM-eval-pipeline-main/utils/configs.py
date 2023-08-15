import re
import numpy as np
from tqdm import tqdm
from transformers import GenerationConfig, AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel
from omegaconf import OmegaConf, DictConfig
from typing import Dict, Tuple, Sequence, List, Union
from transformers import PreTrainedModel, PreTrainedTokenizerFast, PreTrainedTokenizer
import os
import torch
import json


machine = eval(open('machine').read().strip())
config = OmegaConf.load(f'mconfigs/{machine}_config.yaml')



def get_special_loading():  # lookup the specified method to load model and tokenizer. it can be indexed by either model_id (prior) or model_name
    return {
        'llama': load_llama,
        'llama-lora': load_llama_lora,
        'chatglm-6b': load_ChatGLM,
        'doctor-glm': load_DoctorGLM,
    }


def read_config_by_model_id(model_id: str) -> Tuple[str, DictConfig]:
    """Read the config in file by `model_id`, and Also Return the corresponding model_name"""
    if not isinstance(model_id, str):
        raise ValueError(f"model_id should be str, but it is {model_id}")

    for model_name in config.keys():
        if model_id in config[model_name]:
            return model_name, config[model_name][model_id]
    raise NotImplementedError(f"{model_id}, this model id isn't implemented in the `config.yaml`")


def load_system_prompt(model_id: str) -> str:
    """Load and Return prompt by `model_id`"""
    model_name, model_config = read_config_by_model_id(model_id)
    return model_config['prompt']


def load_gconfig(model_id: str, generate_type: str='sample') -> GenerationConfig:
    """Load and Return GenerationConfig for the corresponding `model_id` and `generate_type`"""
    model_name, _ = read_config_by_model_id(model_id)
    return GenerationConfig.from_pretrained("./gconfig", 
                                            config_file_name=f'{generate_type}.json')


def load_model_and_tokenizer(model_id: str) -> Tuple[PreTrainedModel, Union[PreTrainedTokenizerFast, PreTrainedTokenizer]]:
    """Load and Return model and tokenizer by `model_id`"""
    model_name, model_config = read_config_by_model_id(model_id)
    config_dir = model_config['config_dir']
    
    precision = model_config['precision']
    assert precision in ('fp16', ), 'Only supports fp16 for now'

    special_loading = get_special_loading()
    loading_fn = special_loading.get(model_id, special_loading.get(model_name, None))
    if loading_fn:
        return loading_fn(model_id)

    if precision == 'fp16':
        model = AutoModelForCausalLM.from_pretrained(config_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True,
                                                     trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(config_dir, padding_side='left', 
                                              trust_remote_code=True)

    return model, tokenizer








def load_llama(model_id):
    model_name, model_config = read_config_by_model_id(model_id)
    config_dir = model_config['config_dir']

    model = AutoModelForCausalLM.from_pretrained(config_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(config_dir, padding_side='left', use_fast=False, trust_remote_code=True)
    tokenizer.add_special_tokens({
        "eos_token": "<s>",
        "bos_token": "</s>",
        "unk_token": "<unk>",
    })

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(dict(pad_token="<unk>"))

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    return model, tokenizer




def load_llama_lora(model_id):
    from peft import PeftModel
    model_name, model_config = read_config_by_model_id(model_id)
    lora_dir = model_config['lora_dir']

    model, tokenizer = load_llama(model_id)
    model = PeftModel.from_pretrained(model, lora_dir, torch_dtype=torch.float16)
    return model, tokenizer




def load_ChatGLM(model_id):
    model_name, model_config = read_config_by_model_id(model_id)
    config_dir = model_config['config_dir']

    model = AutoModel.from_pretrained(config_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(config_dir, padding_side='left', 
                                              trust_remote_code=True)
    return model, tokenizer


def load_DoctorGLM(model_id):
    model_name, model_config = read_config_by_model_id(model_id)
    config_dir = model_config['config_dir']
    prefix_config_dir = model_config['prefix_config_dir']

    config = AutoConfig.from_pretrained(config_dir, pre_seq_len=128, prefix_projection=False, trust_remote_code=True)
    model = AutoModel.from_pretrained(config_dir, config=config, low_cpu_mem_usage=True, trust_remote_code=True)
    prefix_state_dict = torch.load(prefix_config_dir, map_location=torch.device('cpu'))

    embedding_weight = prefix_state_dict['transformer.prefix_encoder.embedding.weight']
    model.transformer.prefix_encoder.embedding._parameters['weight'] = torch.nn.parameter.Parameter(embedding_weight)
    
    model.half()
    model.transformer.prefix_encoder.float()

    tokenizer = AutoTokenizer.from_pretrained(config_dir, padding_side='left', 
                                              trust_remote_code=True)
    return model, tokenizer










