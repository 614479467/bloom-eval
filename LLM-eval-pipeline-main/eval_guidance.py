from utils.guidance import Guidance_Agent
from benchmark_eval.guidance import *
import argparse

if __name__ == '__main__':
    # TO BE UPDATED

    model_id = 'phoenix-inst-chat-7b'
    batch_size = 1

    agent = Guidance_Agent(model_id, device='cuda')
    Eval = MMCUEvaluation(agent=agent,
                          setting='zero_shot',
                          med_only=False)
    Eval.evaluate()
