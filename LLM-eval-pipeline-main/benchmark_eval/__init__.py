from benchmark_eval.evaluation import (
    MMCUEvaluation,
    MMLUEvaluation,
    GSM8KEvaluation,
)


benchmark2class = {
    'MMCU': MMCUEvaluation,
    'MMLU': MMLUEvaluation,
    'GSM8K': GSM8KEvaluation
}

