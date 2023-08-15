from utils import Agent
from generation import Generation
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--generation_type', type=str, default='greedy')
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()


    agent = Agent(args.model_id, generate_type=args.generation_type)
    Gen = Generation(agent=agent,
                     data_name=args.data_name,
                     batch_size=args.batch_size)
    Gen.generate()

