import numpy as np
import datasets
import os
from typing import Dict, Tuple, Sequence, List, Union



class Environment:
    """
    Load dataset (File: generation/data/$data_name/...)
    Process dataset

    Args:
        data_name (`str`)
            Name of the data, which will be used to find the data file in storage
    """

    def __init__(self, 
                 data_name: str):
        self.data_name = data_name
        self.data_file = os.path.join('generation/data', f'{data_name}.jsonl')

        self.dataset = self.get_dataset(self.data_file)

    def _load_dataset(self, file: str) -> datasets.Dataset:
        """Load data"""
        if not isinstance(file, str):
            raise ValueError(f"file must be str, but is {file}")
        if not (file.endswith('.json') or file.endswith('.jsonl') or file.endswith('.jl')):
            raise ValueError(f"file must be json, but is {file}")

        dataset = datasets.load_dataset("json", data_files=file, split='train')
        print(f'Load in {file}')
        return dataset

    def add_prompt_to_queries(self, queries: List[str], prompt: str='') -> List[str]:
        """Add prompt to queries"""
        if not (isinstance(queries, list) and isinstance(queries[0], str)):
            raise ValueError(f"queries must be List[str], but is {queries}")
        if not isinstance(prompt, str):
            raise ValueError(f"prompt must be str, but is {prompt}")

        return [prompt.format(question=query) for query in queries]

    def _preprocess_dataset(self, dataset: datasets.Dataset) -> datasets.Dataset:
        """Preprocessing"""
        dataset = dataset.map(lambda example: {'processed_query': example['query'].strip()})
        return dataset

    def get_dataset(self, file: str) -> datasets.Dataset:
        """Pipeline"""
        dataset = self._load_dataset(file)
        dataset = self._preprocess_dataset(dataset)
        return dataset




