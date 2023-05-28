# experiment descriptions:
#   1. evaluate all models on downsampled dataset to 150k tokens
#   2. evaluate temporal_attention and UWB models on English up to
#      6.5 million tokens from 250k tokens, with jumps of 250k

import csv
import io
import numpy as np
import os
import random
import torch
from collections.abc import Iterator
from downsample import handle_downsample, sample_random_lines
from evaluate import *
from typing import Callable, Literal

model_names = Literal["UWB", "UG_Student_Intern", "temporal_attention"]

def _reseed(seed: int = None):
    seed = seed if seed is not None else int.from_bytes(os.urandom(16), sys.byteorder)
    
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1)) # limits to ...
    torch.manual_seed(seed % (2**64 - 1)) # limits to ...
    torch.cuda.manual_seed_all(seed)


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
        handle_downsample(language, target, corpora_path)
        print()
        populate_all(corpora_path)

        for model, eval_fn in models_to_eval_fns.items():
            yield (model, eval_fn(language))


def run_stability_experiment(corpora_path: str, results_path: str):
    TARGET_TKNS = 150_000
    EXPS = 260

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

        for language in ["english"]:
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

    token_test_targets = range(250_000, 6_250_000+1, 500_000)
    with open(results_path, "a+") as results_file:
        writer = csv.writer(results_file, delimiter=",")

        results_file.seek(0)
        if results_file.readline().strip("\n") != "Language,Model,Spearman's Rho":
            writer.writerow(["Language", "Model", "Spearman's Rho"])

        for token_target in token_test_targets:
            result_packs = yield_results(corpora_path, token_target, EXPS_PER_TKN_TRGT, "english", fns_pack)
            for model_name, result in result_packs:
                writer.writerow(["english", model_name, result])
                results_file.flush()


def run_experiments():
    corpora_path = "data/downsampled"
    results_dir = "results"
    pathlib.Path(results_dir).mkdir(exist_ok=True, parents=True)

    run_stability_experiment(corpora_path, f"{results_dir}/stability_experiment.csv")
    run_token_size_experiment(corpora_path, f"{results_dir}/token_size_experiment.csv")
    # test_downsample()

    # no_reseed = get_results_dict(corpora_path, 150_000, num_experiments, ign_ta=True)

    # Experiment 2.
    # reseed = get_results_dict(corpora_path, 150_000, num_experiments, True)

    # Experiment 3.
    # token_test_targets = range(250_000, 6_250_000+1, 500_000)

    # results = {
    #     token_target : {
    #         model_name : list of results at token_target
    #     }
    # }
    # results_token_targs: dict[model_names, dict[int, float]] = dict()

    # for token_target in token_test_targets:
    #     results_token_targs[token_target] = get_results_dict(corpora_path, token_target, 10, True)

    # Experiment 4.

    # results = {
    #     language : {
    #         model_name : list of results for model in language
    #     }
    # }
    # pprint(no_reseed)
    # results: dict[str, dict[model_names, list[float]]] = dict()
    # for lang in ["swedish", "german"]:
    #     results[lang] = get_results_dict(corpora_path, 150_000, 10, ign_ta=True, lang=lang)

    # # ENGLISH exps
    # # diff sampled lines
    # print("\n" + "No Reseed Results" + "\n" + "="*50)
    # # pprint(no_reseed)

    # # always same sampled lines, just diff order
    # print("\n" + "Reseeding Results" + "\n" + "="*50)
    # # pprint(reseed)

    # # diff sampled lines, across diff # tokens
    # print("\n" + "Token Target Results" + "\n" + "="*50)
    # # pprint(results_token_targs)

    # # ALL exps
    # print("\n" + "Average Results" + "\n" + "="*50)
    # pprint(results)

if __name__ == "__main__":
    run_experiments()