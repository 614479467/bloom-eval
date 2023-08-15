from utils import Agent
import datasets
import os
import pandas
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str)
    args = parser.parse_args()
    
    model_id = args.model_id


    agent = Agent(model_id, generate_type='alpaca_eval')
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]

    save_set = eval_set.map(lambda x:x).remove_columns(['instruction', 'output', 'generator'])
    save_set = save_set.add_column('instruction',eval_set['instruction'])

    def generate(example):
        example["output"] = agent.response([example["instruction"]])[0].strip()
        example["generator"] = model_id
        return example
    save_set = save_set.map(generate)
    df = pandas.core.frame.DataFrame(save_set.to_list())
    df.to_json(os.path.join("outputs/alpaca_eval", f"{model_id}.json"), orient="records", indent=2)

