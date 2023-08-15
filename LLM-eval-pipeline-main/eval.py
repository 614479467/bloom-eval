from utils import Agent
from benchmark_eval import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--benchmark_name', type=str)
    parser.add_argument('--setting', type=str)
    args = parser.parse_args()

    agent = Agent(args.model_id)
    
    eval_class = benchmark2class[args.benchmark_name]
    Eval = eval_class(agent=agent,
                      batch_size=args.batch_size,
                      setting=args.setting)
    Eval.evaluate()


