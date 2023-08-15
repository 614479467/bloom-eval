import numpy as np
import os
import datasets
from torch.utils.data import DataLoader
import torch
from typing import Dict, Tuple, Sequence, List, Union
import jsonlines
import json
import re
from tqdm import tqdm
import gc
from utils import Agent
from generation.environment import *



class Generation:
    """
    The generation pipeline:
        load preprocessed dataset (from environment classes)
        model generation
        save output files (TO: generation/outputs/$data_name/$model_id.jsonl)
    """
    # output: model_results_dir - generation/outputs/$data_name/$model_id.jsonl
    results_dir = 'generation/outputs'
    def __init__(self, 
                 agent: Agent, 
                 data_name: str,
                 batch_size: int):
        self.agent = agent
        self.data_name = data_name
        self.batch_size = batch_size

        self.model_id = self.agent.model_id
        self.environment = Environment(data_name)
        self.dataset = self.environment.dataset
        
        self.data_results_dir = os.path.join(self.results_dir, self.data_name)
        self.output_file = f'{self.model_id}.jsonl'
        self._create_folder(self.data_results_dir)
    
    def _create_folder(self, path: str):
        """Create a folder for the path if there isn't"""
        if not os.path.exists(path):
            os.makedirs(path)
    
    def _generate_responses(self, dataset: datasets.Dataset, agent: Agent) -> List[str]:
        """Generate responses"""
        if not isinstance(dataset, datasets.Dataset):
            raise ValueError(f"dataset must be datasets.Dataset, but is {dataset}")
        if not isinstance(agent, Agent):
            raise ValueError(f"agent must be Agent, but is {agent}")
        dataloader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=True)
        progress = tqdm(total=len(dataloader))
        response_list = []
        for batch in dataloader:
            responses = agent.response(batch['processed_query'])
            response_list.extend(self._post_process(responses))
            progress.update(1)
        return response_list

    def _formalize_for_saving(self, data_list: List[dict], responses: List[str]) -> List[dict]:
        """Pack the response of each sample with the corresponding query"""
        if not (isinstance(data_list, list) and isinstance(data_list[0], dict)):
            raise ValueError(f"data_list must be List[dict], but is {data_list}")
        if not (isinstance(responses, list) and isinstance(responses[0], str)):
            raise ValueError(f"responses must be List[str], but is {responses}")
        assert len(data_list) == len(responses), f'data_list and responses must have the same length, but are {len(data_list)} and {len(responses)}'

        samples = []
        for data, response in zip(data_list, responses):
            sample = {
                'id': data["id"],
                'label': data["label"],
                'query': data["processed_query"],
            }
            if 'answer' in data:
                sample['answer'] = data['answer']
            sample['output'] = response

            samples.append(sample)
        return samples

    def _post_process(self, text_list: List[str]):
        return [text.strip() for text in text_list]
        
    def _save_output(self, samples: List[dict], save_file: str):
        """Save responses to `$outputs/$data_name/$model_id.jsonl`"""
        if not isinstance(save_file, str):
            raise ValueError(f"save_file must be str, but is {save_file}")
        if not (isinstance(samples, list) and isinstance(samples[0], dict)):
            raise ValueError(f"samples must be List[dict], but is {samples}")
    
        save_path = os.path.join(self.data_results_dir, save_file)
        with jsonlines.open(save_path, 'w') as writer:
            for sample in samples:
                writer.write(sample)
        print(f'Save to {save_path}')

    def generate(self):
        """Pipeline"""
        dataset = self.dataset
        response_list = self._generate_responses(dataset, self.agent)
        result_list = self._formalize_for_saving(dataset.to_list(), response_list)
        self._save_output(result_list, self.output_file)



