import numpy as np
import datasets
import json
import os
from typing import Dict, Tuple, Sequence, List, Union




class BenchmarkBase:
    """
    Loading dataset and few_shot_prompt
    Process dataset for zero_shot, few_shot, or zero_shot_cot setting

    Subclass should config:
        benchmark_name (`str`)
        benchmark_dir (`str`)
        subtasks (`List[str]`)

    Subclass should implement:
        _init_task_prompts(self):
            config task_prompt for each setting (and for each subtask)
    """
    benchmark_name = ''
    benchmark_dir = ''        # benchmark_eval/benchmarks/$benchmark_name/

    subtasks = []

    def __init__(self):
        self._init_task_prompts()

    def _init_task_prompts(self):
        """
        Config self.task_prompt_zero_shot, self.task_prompt_few_shot, and self.task_prompt_zero_shot_cot
        """
        raise NotImplementedError

    @staticmethod
    def _load_jsonl(file: str) -> List[dict]:
        if not isinstance(file, str):
            raise ValueError(f"file must be str, but is {file}")
        if not (file.endswith('.jsonl') or file.endswith('.jl')):
            raise ValueError(f"file must be jsonl, but is {file}")

        with open(file, 'r') as f:
            lines = f.read().strip().split('\n')
            data_list = [json.loads(line) for line in lines]
        return data_list

    # def _load_dataset(self, file: str) -> datasets.Dataset:
    #     """Load one dataset"""
    #     return datasets.Dataset.from_list(self._load_jsonl(os.path.join(self.benchmark_dir, file)))

    def _load_few_shot_prompt(self, file: str) -> str:
        """Load one few_shot_prompt"""
        if not isinstance(file, str):
            raise ValueError(f"file must be str, but is {file}")
        with open(os.path.join(self.benchmark_dir, file)) as f:
            return f.read()

    # def _load_datasets(self):
    #     """Load all datasets"""
    #     if len(self.subtasks):
    #         return {task: self._load_dataset(os.path.join(task, 'test.jsonl')) for task in self.subtasks}
    #     else:
    #         return self._load_dataset('test.jsonl')

    def _load_few_shot_prompts(self):
        """Load all few_shot_prompt"""
        if len(self.subtasks):
            return {task: self._load_few_shot_prompt(os.path.join(task, 'few_shot_prompt')) for task in self.subtasks}
        else:
            return self._load_few_shot_prompt('few_shot_prompt')

    def _add_task_prompt_to_queries(self, queries: List[str], task_prompt: str='') -> List[str]:
        """Add the given task prompt to all queries"""
        if not (isinstance(queries, list) and isinstance(queries[0], str)):
            raise ValueError(f"queries must be List[str], but is {queries}")
        if not isinstance(task_prompt, str):
            raise ValueError(f"task_prompt must be str, but is {task_prompt}")

        return [task_prompt.format(input=query) for query in queries]

    def _add_demonstrations_to_queries(self, queries: List[str], few_shot_prompt: str='') -> List[str]:
        """Add few-shot demonstrations to all queries"""
        if not (isinstance(queries, list) and isinstance(queries[0], str)):
            raise ValueError(f"queries must be List[str], but is {queries}")
        if not isinstance(few_shot_prompt, str):
            raise ValueError(f"few_shot_prompt must be str, but is {few_shot_prompt}")

        return [few_shot_prompt.format(test_question=query) for query in queries]

    def _process_one_dataset_zero_shot(self, dataset: datasets.Dataset, task_prompt: str='') -> datasets.Dataset:
        """Process one dataset for zero-shot setting"""
        return dataset.add_column('prompted_query', self._add_task_prompt_to_queries(dataset['query'], task_prompt))

    def _process_one_dataset_few_shot(self, dataset: datasets.Dataset, few_shot_prompt: str='', task_prompt: str='') -> datasets.Dataset:
        """Process one dataset for few-shot setting"""
        prompted_ICL = self._add_demonstrations_to_queries(dataset['query'], few_shot_prompt)
        return dataset.add_column('prompted_query', self._add_task_prompt_to_queries(prompted_ICL, task_prompt))
    
    def _process_datasets(self, 
                          datasets: Union[datasets.Dataset, Dict[str, datasets.Dataset]],
                          few_shot_prompts: Union[str, Dict[str, str]],
                          setting: str,
                         ) -> Union[datasets.Dataset, Dict[str, datasets.Dataset]]:
        """Process all datasets, available for zero_shot, few_shot, zero_shot_cot settings"""

        if not (setting in ('zero_shot', 'few_shot', 'zero_shot_cot')):
            raise ValueError(f"setting should be in ('zero_shot', 'few_shot', 'zero_shot_cot'), but it is {setting}")
        assert (isinstance(datasets, dict) and isinstance(few_shot_prompts, dict)) or (not isinstance(datasets, dict) and not isinstance(few_shot_prompts, dict))

        if isinstance(datasets, dict):
            assert set(datasets.keys()) == set(few_shot_prompts.keys())
            processed_datasets = dict()
            for name in datasets.keys():
                if setting == 'few_shot':
                    processed_datasets[name] = self._process_one_dataset_few_shot(datasets[name], few_shot_prompts[name], self.task_prompt_few_shot[name])
                elif setting == 'zero_shot':
                    processed_datasets[name] = self._process_one_dataset_zero_shot(datasets[name], self.task_prompt_zero_shot[name])
                else:
                    processed_datasets[name] = self._process_one_dataset_zero_shot(datasets[name], self.task_prompt_zero_shot_cot[name])
            return processed_datasets

        else:
            if setting == 'few_shot':
                processed_dataset = self._process_one_dataset_few_shot(datasets, few_shot_prompts, self.task_prompt_few_shot)
            elif setting == 'zero_shot':
                processed_dataset = self._process_one_dataset_zero_shot(datasets, self.task_prompt_zero_shot)
            else:
                processed_dataset = self._process_one_dataset_zero_shot(datasets, self.task_prompt_zero_shot_cot)
            return processed_dataset

    def get(self, setting) -> Tuple[Union[datasets.Dataset, Dict[str, datasets.Dataset]], str]:
        """Load and Process all datasets"""
        all_datasets: Union[datasets.Dataset, Dict[str, datasets.Dataset]] = self._prepare_data()
        if setting == 'few_shot':
            few_shot_prompts: Union[str, Dict[str, str]] = self._load_few_shot_prompts()
        else:
            if isinstance(all_datasets, dict):
                few_shot_prompts = {name: '' for name in all_datasets}
            else:
                few_shot_prompts = ''

        processed_datasets = self._process_datasets(all_datasets, few_shot_prompts, setting)
        return processed_datasets



class MMCUBenchmark(BenchmarkBase):
    # w/o zero_shot_cot
    benchmark_name = 'MMCU'
    benchmark_dir = os.path.join('benchmark_eval/benchmarks', 'MMCU', 'converted')

    top_subjects = ['心理', '医疗', '教育', '法律']
    subtasks = ['医疗_临床医学', '医疗_传染病学', '医疗_儿科学', '医疗_免疫学', '医疗_医学三基', '医疗_医学影像学', '医疗_外科学', '医疗_寄生虫学', '医疗_护理学', '医疗_病理学', '医疗_皮肤性病学', '医疗_组织胚胎学', '医疗_药物分析学', '医疗_药理学', '医疗_解剖学', 
                '教育_化学', '教育_历史', '教育_地理', '教育_政治', '教育_数学', '教育_物理', '教育_生物', '教育_语文', 
                '法律',
                '心理']

    def __init__(self):
        super(MMCUBenchmark, self).__init__()

    def _init_task_prompts(self):
        """
        Config self.task_prompt_zero_shot, self.task_prompt_few_shot, and self.task_prompt_zero_shot_cot
        """
        task_prompt = '请阅读以下选择题并给出正确选项，不要解释原因。请只给出答案的序号。\n{input}'
        self.task_prompt_zero_shot = self.task_prompt_few_shot = {
            task: task_prompt for task in self.subtasks
        }

class MMLUBenchmark(BenchmarkBase):
    # w/o zero_shot_cot
    benchmark_name = 'MMLU'
    benchmark_dir = os.path.join('benchmark_eval/benchmarks', 'MMLU', 'converted')

    top_subjects = ['Humanities', 'Social Science', 'STEM', 'Other']
    subtasks = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security',
                'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history',
                'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history',
                'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 
                'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']

    def __init__(self):
        super(MMLUBenchmark, self).__init__()

    def _init_task_prompts(self):
        """
        Config self.task_prompt_zero_shot, self.task_prompt_few_shot, and self.task_prompt_zero_shot_cot
        """
        self.task_prompt_zero_shot = self.task_prompt_few_shot = {
            task: 'The following are multiple choice questions (with answers) about %s.\n\n{input}' % (' '.join(task.split('_'))) for task in self.subtasks
        }

class GSM8KBenchmark(BenchmarkBase):
    benchmark_name = 'GSM8K'
    # benchmark_dir = os.path.join('benchmark_eval/benchmarks', 'GSM8K', 'converted')
    benchmark_dir = os.path.join('benchmark_eval/benchmarks', 'GSM8K', 'origin')

    def __init__(self):
        super(GSM8KBenchmark, self).__init__()

    def _init_task_prompts(self):
        """
        Config self.task_prompt_zero_shot, self.task_prompt_few_shot, and self.task_prompt_zero_shot_cot
        """
        self.task_prompt_few_shot = '{input}'
        self.task_prompt_zero_shot = 'Question: {input}\n'
        self.task_prompt_zero_shot_cot = 'Question: {input}\n'
        
    def _prepare_data(self):
        data_list = self._load_jsonl(os.path.join(self.benchmark_dir, 'test.jsonl'))

        processed_data_list = []
        for i, data in enumerate(data_list):
            processed_data_list.append({
                'query_id': i,
                'query': data['question'],
                'answer': data['answer'].split('#### ')[-1].strip(),
            })
        return datasets.Dataset.from_list(processed_data_list)





