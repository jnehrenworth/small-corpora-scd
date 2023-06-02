# experiment descriptions:
#   1. evaluate all models on downsampled dataset to 150k tokens
#   2. evaluate temporal_attention and UWB models on English up to
#      6.5 million tokens from 250k tokens, with jumps of 250k

import csv
import numpy as np
import os
import random
import torch
from collections.abc import Iterator
from downsample import handle_downsample
from evaluate import *
from typing import Callable, Literal

model_names = Literal["UWB", "UG_Student_Intern", "temporal_attention"]

def _reseed(seed: int = None):
    seed = seed if seed is not None else int.from_bytes(os.urandom(16), sys.byteorder)
    
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1)) # limits to ...
    torch.manual_seed(seed % (2**64 - 1)) # limits to ...
    torch.cuda.manual_seed_all(seed % (2**64 - 1)) # limits to ...


def yield_results(
    corpora_path: str,
    target: int, 
    iters: int, 
    language: str,
    models_to_eval_fns: dict[model_names, Callable],
    seed: int = None,
) -> Iterator[tuple[model_names, int]]:

    for i in range(iters):
        _reseed(seed)

        print("\n" + "="*25 + f" Evaluation # {i+1} " + "="*25 + "\n")
        handle_downsample(language, target, corpora_path, disable_warn=True)
        print()
        populate_all(corpora_path)

        for model, eval_fn in models_to_eval_fns.items():
            yield (model, eval_fn(language))


def run_stability_experiment(corpora_path: str, results_path: str):
    TARGET_TKNS = 150_000
    EXPS = 6

    fns_pack = {
        'UWB': evaluation_rules_UWB,
        'UG_Student_Intern': evaluation_rules_UGSI,
        'temporal_attention': evaluation_rules_TA
    }

    with open(results_path, "a+") as results_file:
        writer = csv.writer(results_file, delimiter=",")
        
        results_file.seek(0)
        if results_file.readline().strip("\n") != "Language,Model,Spearman's Rho":
            writer.writerow(["Language", "Model", "Spearman's Rho"])

        for language in ["swedish"]:
            result_packs = yield_results(corpora_path, TARGET_TKNS, EXPS, language, fns_pack)
            for model_name, result in result_packs:
                writer.writerow([language, model_name, result])
                results_file.flush()


def run_token_size_experiment(corpora_path: str, results_path: str):
    EXPS_PER_TKN_TRGT = 10

    fns_pack = {
        'UWB': evaluation_rules_UWB,
        'UG_Student_Intern': evaluation_rules_UGSI,
        'temporal_attention': evaluation_rules_TA
    }

    token_test_targets = range(5_750_000, 6_250_000+1, 500_000)
    with open(results_path, "a+") as results_file:
        writer = csv.writer(results_file, delimiter=",")

        results_file.seek(0)
        if results_file.readline().strip("\n") != "Language,Model,Tokens,Spearman's Rho":
            writer.writerow(["Language", "Model", "Tokens", "Spearman's Rho"])

        for model_name, result in yield_results(corpora_path, 5250000, 1, "english", {"temporal_attention": evaluation_rules_TA}):
            writer.writerow(["english", model_name, 5250000, result])

        for model_name, result in yield_results(corpora_path, 5250000, 1, "english", fns_pack):
            writer.writerow(["english", model_name, 5250000, result])

        for token_target in token_test_targets:
            result_packs = yield_results(corpora_path, token_target, EXPS_PER_TKN_TRGT, "english", fns_pack)
            for model_name, result in result_packs:
                writer.writerow(["english", model_name, token_target, result])
                results_file.flush()


def run_experiments():
    corpora_path = "data/downsampled"
    results_dir = "results"
    Path(results_dir).mkdir(exist_ok=True, parents=True)

    # run_stability_experiment(corpora_path, f"{results_dir}/stability_experiment.csv")
    run_token_size_experiment(corpora_path, f"{results_dir}/token_size_experiment.csv")

if __name__ == "__main__":
    run_experiments()
