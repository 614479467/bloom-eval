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
from benchmark_eval.data_utils import *



class EvaluationBase:
    """
    The evaluation pipeline:
        load datasets (from benchmark classes)
        model generation and decoding
        metrics calculation
        save output files and metrics file

    Subclass should config:
        benchmark (`BenchmarkBase`)

    Subclass should implement:
        _decode_one_response(self, response: str):
            decode the answer, especially for the objective questions

        _compute_metrics(self, samples: List[dict], task_name: str):
            specify what metrics to compute for different subtasks

        _aggregate_metrics(self, metrics: Dict[str, Dict[str, float]], lens: Dict[str, int]):
            specify whether and how to aggregate metrics over different subtasks
    """
    # output: model_results_dir - benchmark_eval/results/$setting/$benchmark_name/$model_id
    results_dir = 'benchmark_eval/results'
    def __init__(self, 
                 agent: Agent, 
                 benchmark: BenchmarkBase,
                 batch_size: int,
                 setting: str='zero_shot'):
        self.agent = agent
        self.benchmark = benchmark
        self.batch_size = batch_size
        self.setting = setting

        if not (setting in ('zero_shot', 'few_shot', 'zero_shot_cot')):
            raise ValueError(f"setting should be in ('zero_shot', 'few_shot', 'zero_shot_cot'), but it is {setting}")

        self.benchmark_name = benchmark.benchmark_name
        self.subtasks = benchmark.subtasks

        self.model_id = self.agent.model_id
        self.setting_results_dir = os.path.join(self.results_dir, self.setting)
        self.benchmark_results_dir = os.path.join(self.setting_results_dir, self.benchmark_name)
        self.model_results_dir = os.path.join(self.benchmark_results_dir, self.model_id)
        self._create_folder(self.setting_results_dir)
        self._create_folder(self.benchmark_results_dir)
        self._create_folder(self.model_results_dir)
    
    def _create_folder(self, path: str):
        """Create a folder for the path if there isn't"""
        if not os.path.exists(path):
            os.makedirs(path)

    def _decode_one_response(self, response: str) -> str:
        """decode one response to derive the required answer"""
        if not isinstance(response, str):
            raise ValueError(f"response must be str, but is {response}")
        raise NotImplementedError

    def _decode_responses(self, responses: List[str]) -> List[str]:
        """decode all responses to derive the required answers"""
        if not (isinstance(responses, list) and isinstance(responses[0], str)):
            raise ValueError(f"responses must be List[str], but is {responses}")
        return [self._decode_one_response(response) for response in responses]
    
    def _zip_qra(self, dataset: List[dict], responses: List[str], ex_responses: List[str]) -> List[dict]:
        """Pack the response and its decoded answer of each sample with the corresponding query and answer"""
        assert len(dataset) == len(responses) == len(ex_responses), f'dataset, responses and ex_responses must have the same length, but are {len(dataset)}, {len(responses)} and {len(ex_responses)}'
        samples = []
        for data, response, ex_response in zip(dataset, responses, ex_responses):
            samples.append({
                **data,
                'response': response,
                'response_answer': ex_response,
            })
        return samples
    
    def save_response(self, file: str, samples: List[dict]):
        """Save responses to `$results/$benchmark_name/$model_id/file`"""
        if not isinstance(file, str):
            raise ValueError(f"file must be str, but is {file}")
        if not (isinstance(samples, list) and isinstance(samples[0], dict)):
            raise ValueError(f"samples must be List[dict], but is {samples}")
    
        with jsonlines.open(os.path.join(self.model_results_dir, file), 'w') as writer:
            for sample in samples:
                writer.write(sample)

    def save_metrics(self, metrics: dict):
        """Save metrics to `$results/$benchmark_name/$model_id/metrics.json`"""
        if not isinstance(metrics, dict):
            raise ValueError(f"metrics must be dict, but is {dict}")
    
        with open(os.path.join(self.model_results_dir, 'metrics.json'), 'w') as f:
            f.write(json.dumps(metrics, indent=4, ensure_ascii=False) + '\n')

    def compute_acc(self, ex_responses: list, answers: list) -> float:
        """Compute accuracy with extracted responses and ground answers"""
        if not (isinstance(ex_responses, list) and isinstance(answers, list)):
            raise ValueError(f"ex_responses and answers must be both list, but is {ex_responses} and {answers}")
        return np.mean([r == a for r, a in zip(ex_responses, answers)])

    def _compute_metrics(self, samples: List[dict], task_name: str):
        """
        Compute all metrics for samples. You can compute different metrics for different tasks
        Args:
            samples (`List[dict]`):
                List of samples
            task_name (`str`)

        Return:
            metrics (`Dict[str, float]`):
                {
                    'metric_name': metric
                }
        """
        raise NotImplementedError
    
    def _aggregate_metrics(self, metrics: Dict[str, Dict[str, float]], lens: Dict[str, int]) -> Dict[str, Dict[str, float]]:
        """
        Aggregate metrics for subtasks
        Args:
            metrics (`Dict[str, Dict[str, float]]`): 
                {'task_name': {'metric_name': metric}}
            lens (`Dict[str, int]`): 
                {'task_name': #task_examples}
        """
        raise NotImplementedError

    def _count_examples(self, datasets: Union[datasets.Dataset, Dict[str, datasets.Dataset]]) -> Dict[str, int]:
        if len(self.subtasks):
            return {subtask: len(datasets[subtask]) for subtask in self.subtasks}
        else:
            return {self.benchmark_name: len(datasets)}

    def _eval_one_dataset(self, 
                          subtask_name: str, 
                          dataset: datasets.Dataset) -> Dict[str, float]:
        """
        Evaluate and return the metrics for one dataset
        Args:
            subtask_name (`str`)
            dataset (`datasets.Dataset`)
        Return:
            metrics (`Dict[str, float]`):
                {
                    'metric_name': metric
                }
        """
        if not isinstance(subtask_name, str):
            raise ValueError(f"subtask_name must be str, but is {subtask_name}")
        if not isinstance(dataset, datasets.Dataset):
            raise ValueError(f"dataset must be datasets.Dataset, but is {dataset}")

        dataloader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=True)
        response_list = []
        for batch in tqdm(dataloader, desc=f'{subtask_name}'):
            responses: List[str] = self.agent.get_responses(batch['prompted_query'], self.setting)
            response_list.extend(responses)

        ex_response_list = self._decode_responses(response_list)

        samples = self._zip_qra(dataset.to_list(), response_list, ex_response_list)
        self.save_response(f"{subtask_name}.jsonl", samples)
        metrics = self._compute_metrics(samples, subtask_name)
        gc.collect(); torch.cuda.empty_cache()
        print(metrics)
        return metrics

    def _eval_datasets(self,
                       all_datasets: Union[datasets.Dataset, Dict[str, datasets.Dataset]]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate and return the metrics for all datasets
        Return:
            metrics (`Dict[str, dict]`):
                {
                    'task_name': {
                            'metric_name': metric
                        }
                }
        """
        if len(self.subtasks):
            return {subtask: self._eval_one_dataset(subtask, all_datasets[subtask]) for subtask in self.subtasks}
        else:
            return {self.benchmark_name: self._eval_one_dataset(self.benchmark_name, all_datasets)}

    def evaluate(self):
        all_datasets = self.benchmark.get(self.setting)
        lens = self._count_examples(all_datasets)
        metrics = self._eval_datasets(all_datasets)
        aggr_metrics = self._aggregate_metrics(metrics, lens)
        self.save_metrics(aggr_metrics)


class MMCUEvaluation(EvaluationBase):
    # output: model_results_dir - benchmark_eval/results/$setting/MMCU/$model_id
    
    def __init__(self, 
                 agent: Agent, 
                 batch_size: int,
                 setting: str='zero_shot'):
        
        super(MMCUEvaluation, self).__init__(agent=agent, 
                                             batch_size=batch_size,
                                             setting=setting,
                                             benchmark=MMCUBenchmark())

    def _decode_one_response(self, response: str) -> str:
        """decode option(s) from one response and standardize it"""
        if not isinstance(response, str):
            raise ValueError(f"response must be str, but is {response}")

        def standardize_options(options):
            return ''.join(sorted(set(options)))
        return standardize_options(re.findall(r'[ABCD]', response))
    
    def _compute_metrics(self, samples: List[dict], task_name: str):
        ex_responses = [sample['response_answer'] for sample in samples]
        answers = [sample['answer'] for sample in samples]
        return {
            'Accuracy': self.compute_acc(ex_responses, answers)
        }

    def _aggregate_metrics(self, metrics: Dict[str, Dict[str, float]], lens: Dict[str, int]) -> Dict[str, Dict[str, float]]:
        """
        Aggregate medicine and eduction
        """
        aggr_metrics = {}
        metric_names = list(metrics.values())[0].keys()

        def aggregate(top_subject):
            aggr_metrics[top_subject] = {}
            subjects = list(filter(lambda x: x.startswith(f'{top_subject}_'), self.subtasks))
            for subject in subjects:
                aggr_metrics[top_subject][subject] = metrics[subject]

            aggr_metrics[top_subject]['average'] = {}
            aggr_metrics[top_subject]['overall'] = {}
            for metric_name in metric_names:
                aggr_metrics[top_subject]['average'][metric_name] = np.mean([metrics[subject][metric_name] for subject in subjects])
                aggr_metrics[top_subject]['overall'][metric_name] = np.sum([lens[subject] * metrics[subject][metric_name] for subject in subjects]) / np.sum([lens[subject] for subject in subjects])

        aggregate('医疗')
        aggregate('教育')
        aggr_metrics['法律'] = metrics['法律']
        aggr_metrics['心理'] = metrics['心理']

        return aggr_metrics


class MMLUEvaluation(EvaluationBase):
    # output: model_results_dir - benchmark_eval/results/$setting/MMCU/$model_id
    
    def __init__(self, 
                 agent: Agent, 
                 batch_size: int,
                 setting: str='zero_shot'):
        
        super(MMLUEvaluation, self).__init__(agent=agent, 
                                             batch_size=batch_size,
                                             setting=setting,
                                             benchmark=MMLUBenchmark())

    def _decode_one_response(self, response: str) -> str:
        """decode option(s) from one response and standardize it"""
        if not isinstance(response, str):
            raise ValueError(f"response must be str, but is {response}")

        def standardize_options(options):
            return ''.join(sorted(set(options)))
        return standardize_options(re.findall(r'[ABCD]', response))
    
    def _compute_metrics(self, samples: List[dict], task_name: str):
        ex_responses = [sample['response_answer'] for sample in samples]
        answers = [sample['answer'] for sample in samples]
        return {
            'Accuracy': self.compute_acc(ex_responses, answers)
        }

    def _aggregate_metrics(self, metrics: Dict[str, Dict[str, float]], lens: Dict[str, int]) -> Dict[str, Dict[str, float]]:
        """
        Aggregate 'Humanities', 'Social Science', 'STEM', 'Other'
        """
        aggr_metrics = {}
        metric_names = list(metrics.values())[0].keys()

        def aggregate(top_subject):
            aggr_metrics[top_subject] = {}
            task_lst = self.subtasks
            STEM_lst = []            
            hum_lst = []
            soc_list = []            
            
            for task in self.subtasks:
                # STEM
                if (task == 'abstract_algebra'| task == 'anatomy' | task == 'astronomy' | task.contains('biology') | task.contains('chemistry') | task.contains('computer_science') | task.contains('mathematics') | task.contains('physics') |
                    task.contains('computer_security') | task.contains('electrical_engineering') | task.contains('statistics') | task.contains('machine_learning') | task.contains('machine_learning')):
                    STEM_lst.append(task)
                    task_lst.remove(task)

                # 'Humanities'
                elif (task == 'world_religions'| task == 'jurisprudence' | task == 'philosophy' | task.contains('history') | task.contains('logic') | task.contains('moral') | task.contains('law')):
                    hum_lst.append(task)
                    task_lst.remove(task)
                # Social Science
                elif (task == 'high_school_geography'| task == 'human_sexuality' | task == 'sociology' | task == 'public_relations' | task == 'security_studies' | task.contains('econom') | task.contains('poli') | task.contains('psychology')):
                    hum_lst.append(task)
                    task_lst.remove(task)            
                # Other
                else:
                    other_lst = task_lst

            if top_subject == 'STEM':
                subjects = STEM_lst
            elif top_subject == 'Humanities':
                subjects = hum_lst
            elif top_subject == 'Social Science':
                subjects = soc_list
            else:
                subjects = other_lst

            for subject in subjects:
                aggr_metrics[top_subject][subject] = metrics[subject]

            aggr_metrics[top_subject]['average'] = {}
            aggr_metrics[top_subject]['overall'] = {}
            for metric_name in metric_names:
                aggr_metrics[top_subject]['average'][metric_name] = np.mean([metrics[subject][metric_name] for subject in subjects])
                aggr_metrics[top_subject]['overall'][metric_name] = np.sum([lens[subject] * metrics[subject][metric_name] for subject in subjects]) / np.sum([lens[subject] for subject in subjects])

        aggregate('Humanities')
        aggregate('Social Science')
        aggregate('STEM')
        aggregate('Other')


        return aggr_metrics
        
class GSM8KEvaluation(EvaluationBase):
    # output: model_results_dir - benchmark_eval/results/$setting/GSM8K/$model_id
    
    def __init__(self, 
                 agent: Agent, 
                 batch_size: int,
                 setting: str='zero_shot_cot'):
        
        super(GSM8KEvaluation, self).__init__(agent=agent, 
                                             batch_size=batch_size,
                                             setting=setting,
                                             benchmark=GSM8KBenchmark())

    def _decode_one_response(self, response: str) -> str:
        """decode numerical value from one response"""
        if not isinstance(response, str):
            raise ValueError(f"response must be str, but is {response}")

        pattern = r'(?:[\s=+/<>(:€\$\.\-\*\\])(?=\S)((?:0|(?:[1-9](?:\d*|\d{0,2}(?:,\d{3})*)))?(?:\.\d+)?(?:%|g|kg|kgs|"|st|nd|rd|th|cm|m|mm)?)(?:(?![^\s=+/>)$:\.\-\*\\])|(?=, ))'

        pred = re.search(r'[Tt]he answer is(.+)', response)
        pred = pred.group(1) if pred else response
        pred = list(filter(lambda x: len(x.strip()) != 0 and x != '%', re.findall(pattern, pred)))
        pred = pred[-1].strip() if pred else ''

        return self._standardize(pred)
        
    def _compute_metrics(self, samples: List[dict], task_name: str):
        ex_responses = [sample['response_answer'] for sample in samples]
        answers = [self._standardize(str(sample['answer'])) for sample in samples]
        return {
            'Accuracy': self.compute_acc(ex_responses, answers)
        }

    def _standardize(self, x):
        """Standardize numerical values"""
        if not len(x):
            return ''
        y = ''.join(x.split(','))
        if '.' in y:
            y = y.rstrip('0')
            if y[-1] == '.':
                y = y[:-1]
        if y[0] == '.':
            y = '0' + y
        if y[-1] == '%':
            y = str(eval(y[:-1]) / 100)
        return y.rstrip('kgs').rstrip('kg').rstrip('g').rstrip('"').rstrip('st').rstrip('nd').rstrip('rd').rstrip('th').rstrip('cm').rstrip('mm').rstrip('m')

    def _aggregate_metrics(self, metrics: Dict[str, Dict[str, float]], lens: Dict[str, int]) -> Dict[str, Dict[str, float]]:
        return metrics


