import re
import numpy as np
from tqdm import tqdm
from transformers import GenerationConfig, PreTrainedModel
from omegaconf import OmegaConf, DictConfig
from typing import Dict, Tuple, Sequence, List, Union
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer
from retrying import retry
import os
import torch
import gc
import json
import logging
from utils.configs import load_gconfig, load_model_and_tokenizer, load_system_prompt



class Agent:
    def __init__(self, 
                 model_id: str, 
                 generate_type: str='greedy',
                 device: str='cuda'):
        self.model_id = model_id
        self.device = device
        self.generate_type = generate_type

        self.model, self.tokenizer = load_model_and_tokenizer(model_id)
        self.system_prompt = load_system_prompt(model_id)
        self.gconfig = load_gconfig(model_id, generate_type=generate_type)

        self.model.eval().to(self.device)

    def from_model(self, 
                   model: PreTrainedModel, 
                   tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer], 
                   system_prompt: str,
                   generate_type: str='sample'):
        self.model, self.tokenizer = model, tokenizer
        self.system_prompt = system_prompt
        self.gconfig = load_gconfig(model_id, generate_type=generate_type)

        self.device = model.device
        self.model.eval()

    @retry(wait_fixed=10000, stop_max_attempt_number=5)
    def get_responses(self, query_list: List[str], setting: str) -> List[str]:
        """
        Sample one response for each query
        Args:
            query_list (`List[str]`)
                List of queries
            setting (`str`)
        Return:
            responses (`List[str]`)
                List of responses to the corresponding queries
        """
        if not (isinstance(query_list, list) and isinstance(query_list[0], str)):
            raise ValueError(f"query_list should be List[str], but it is {query_list}")
        if not isinstance(setting, str):
            raise ValueError(f"setting should be str, but it is {setting}")
        
        if setting == 'zero_shot_cot':
            return self._response_zero_shot_cot(query_list)
        else:
            return self._response(query_list)

    @torch.inference_mode(mode=True)
    def _response(self, query_list: List[str]) -> List[str]:
        

        query_list = [self.system_prompt.format(question=query) for query in query_list]
        input_ids = self.tokenizer(query_list, padding=True, truncation=True, return_tensors="pt").input_ids.to(self.device)
        n_seq = input_ids.shape[-1]

        output_ids = self.model.generate(input_ids=input_ids, 
                                         generation_config=self.gconfig
                                         )[..., n_seq:]
        responses = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return responses

    @torch.inference_mode(mode=True)
    def _response_zero_shot_cot(self, query_list: List[str]) -> List[str]:
        """
        Sample one response for each query, compatible with zero-shot, few-shot, and zero-shot cot setting
        Args:
            query_list (`List[str]`)
                List of queries
        Return:
            responses (`List[str]`)
                List of responses to the corresponding queries
        """
        
        if not (isinstance(query_list, list) and isinstance(query_list[0], str)):
            raise ValueError(f"query_list should be List[str], but it is {query_list}")

        query_list = [self.system_prompt.format(question=query) + "Let's think step by step." for query in query_list]
        input_ids = self.tokenizer(query_list, padding=True, truncation=True, return_tensors="pt").input_ids.to(self.device)
        n_seq = input_ids.shape[-1]

        output_ids = self.model.generate(input_ids=input_ids, 
                                         generation_config=self.gconfig
                                         )[..., n_seq:]
        responses = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return responses


