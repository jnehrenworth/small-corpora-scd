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


#########################################
# Helpers                               #
#########################################

def _reseed(seed: int = None):
    """Seeds random, np, and torch with `seed`, if provided, otherwise seeds
    them based on system entropy

    If no seed is provided, this function works identically to the way random.seed()
    sets its seed if no argument is provided. 
    """
    # If you do enough digging into how the python random.seed() method works if no
    # seed is provided, eventually you will uncover that it first tries to use system entropy
    # similar to os.urandom (see https://docs.python.org/3.12/library/random.html?highlight=random#random.seed).
    # Digging through the documentation to find what this high-level aspiration translates
    # to at a source-code level, eventually you arrive at the following C code as the
    # ultimate point where the random seed is set: https://github.com/python/cpython/blob/3.12/Modules/_randommodule.c#L245.
    # This roughly translates to the following Python line
    seed = seed if seed is not None else int.from_bytes(os.urandom(16), sys.byteorder)
    
    random.seed(seed)

    # Each of the following seed-setting functions will error if a number larger than
    # 2^32-1 (for np) or 2^64-1 (for torch) is provided
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed % (2**64 - 1))
    torch.cuda.manual_seed_all(seed % (2**64 - 1))


def yield_results(
    corpora_path: str,
    target: int, 
    iters: int, 
    language: str,
    models_to_eval_fns: dict[model_names, Callable[[str], float]],
    seed: int = None,
) -> Iterator[tuple[model_names, float]]:
    """Yields the results of evaluating the models in `models_to_eval_fns.keys()` on
    newly downsampled SemEval corpora for `language` `iters` number of times.

    This function is the main driver of all experiments presented in our paper.  Informally,
    this function just provides a generator of tuples of the form (`model_name`, Spearman's rho).
    Slightly more exactly, this function repeats the following steps `iters` number of times: 

        1. Downsample the SemEval corpora given by `language` to around `target` number of tokens,
        2. Save these downsampled corpora to `corpora_path`, and
        3. Evaluate each model in `models_to_eval_fns.keys()` on the downsampled corpora
           and yield these results in tuple form.
    
    Before this function is used, there must be a directory structure in the following form:

        1. Annotated Uses: data/annotated_uses/{language}/{target_word}.csv
        2. SemEval 2020-Task 1 Corpora: data/semeval/{language}/{corpus_name}.txt
        3. SemEval 2020-Task 1 Targets: data/semeval/{language}/targets.txt

    The optional seed function can be used for reproducibility or when debugging to seed
    random, np, and torch.
    
    Parameters
    ----------
    `corpora_path` : str
        The path to save the downsampled corpora created with this function.
    `target`: int
        Roughly how many tokens should be in the downsampled corpora.
    `iters` : str
        How many times to run the downsample evaluate loop.
    `language` : str
        One of "english", "swedish", or "german".  Must be lowercase.
    `models_to_eval_fns` : dict[model_names, Callable[[str], float]],
        A mapping of the name of each model to be evaluated to a function
        that can be used to evaluate that model.  This function should take 
        a language in {"english", "german", "swedish"} (can assume lowercase) 
        and return a float representing Spearman's rho calculated by training 
        the model on that language and then comparing the model's predicted 
        performance against the ground truth ranking.  
        
        See `evaluation_rules_UGSI` for an example of how this can be done.

        The function can expect that all relevant datapaths have previously been 
        populated, so that the only role of the function should be to train the 
        model on the corpora of a given language, predict ranked change, and 
        then return Spearman's rho comparing predicted to ground truth rankings.

        It can be nice if the `evaluation_rules` function has some pretty printing
        that keeps the user appraised of training progress, but of course that is
        not necessary.
    `seed` : int
        Random seed to use, if None then system random is used.
    """
    for i in range(iters):
        # It may seem strange to reseed on each iteration of this loop rather than seeding
        # once and being done with it.  The reason we do this is that somehow the 
        # temporal_attention model is resetting the seed each time it is run.  Despite
        # searching through the temporal_attention code-base and removing all  
        # instances of a seed being set, this behaviour persists.  It's possible that one
        # of the dependencies it calls is automatically setting a seed.  Obviously, this
        # isn't desirable behaviour because it means that we end up always choosing the
        # same random lines in the downsampling process.  To get around this problem, we manually
        # reseed on each experiment iteration.
        _reseed(seed)

        print("\n" + "="*25 + f" Evaluation # {i+1} " + "="*25 + "\n")
        
        # disable_warn is necessary to pass in because there are 156k tokens that must be included
        # in one of the downsampled German corpora.  This is greater than the 150k tokens
        # aimed for by run_stability_experiment so would normally prompt a message asking the 
        # user to confirm they want to continue with the downsample.  It becomes rather onerous
        # when running many experiments to manually confirm, so instead we just disable warnings
        # altogether.  In some ways, this couples this function to run_stability_experiment, but
        # it also seems as if it is generally desired behaviour when running any token experiment.
        # 
        # If you are creating your own experiments, consider removing disable_warn during the 
        # debugging / experiment building stage.
        handle_downsample(language, target, corpora_path, disable_warn=True)
        print() # just for nicer spacing
        populate_all(corpora_path)

        for model, eval_fn in models_to_eval_fns.items():
            yield (model, eval_fn(language))

#########################################
# Experiments                           #
#########################################

def run_stability_experiment(corpora_path: str, results_path: str):
    """Downsamples the SemEval-2020 Task 1 corpora five hundred different times to 150k tokens,
    writing to `corpora_path` the downsamples, and evaluates each of the UWB,
    UG_Student_Intern, and temporal_attention models on these downsampled corpora,
    saving the results to a csv given by `results_path`.

    This function was used to generate the results of experiments 5.1 and 5.2 
    in our paper.  The results_path file must be a .csv but it need not previously exist.
    If it does, results will be appended to it in the form: language, model_name, Spearman's rho.
    If it does not, it will be created.  The parent directory to the .csv file, however, 
    must exist before this function is used.
    
    Note that the following code can be used to map model names to author names, assuming
    that one wishes to use pandas to read in the results csv:
    ```
    results_path = "results/stability_experiment.csv"

    df = pd.read_csv(results_path)
    df.replace("UG_Student_Intern", "Pömsl and Lyapin (2020)", inplace=True)
    df.replace("UWB", "Pražák et al. (2020)", inplace=True)
    df.replace("temporal_attention", "Rosin and Radinsky (2022)", inplace=True)
    ```
    """
    TARGET_TKNS = 150_000
    EXPS = 500

    # Add a new model here!  See `evaluate.py` for more information on adding a new model
    # If you wish to run the exact experiments we did on these additional models,
    # we recommend commenting out the UWB, UGSI, and TA models in the fns_pack definition
    # below and instead provide maps of just the additional models you've added.  If you do
    # this, it will append to the results_path .csv file results from these additional models.
    # In this way, no information gets overwritten and you don't have to duplicate computationally
    # expensive experiments on UWB, UGSI, and TA.
    fns_pack = {
        'UWB': evaluation_rules_UWB,
        'UG_Student_Intern': evaluation_rules_UGSI,
        'temporal_attention': evaluation_rules_TA
    }

    with open(results_path, "a+") as results_file:
        writer = csv.writer(results_file, delimiter=",")
        
        # The results file may already exist, but, then again, it may not (this is why
        # we open it in a+ mode).  To account for either possibility we have to determine
        # whether it's necessary to write a csv heading row or no.
        results_file.seek(0)
        if results_file.readline().strip("\n") != "Language,Model,Spearman's Rho":
            writer.writerow(["Language", "Model", "Spearman's Rho"])

        for language in ["english", "german", "swedish"]:
            result_packs = yield_results(corpora_path, TARGET_TKNS, EXPS, language, fns_pack)
            for model_name, result in result_packs:
                writer.writerow([language, model_name, result])
                results_file.flush() 


def run_token_size_experiment(corpora_path: str, results_path: str):
    """Downsamples the SemEval-2020 Task 1 English corpora fifty different times to token 
    targets from 250k to 6.25M, with jumps of 500k tokens and evaluates each of the UWB,
    UG_Student_Intern, and temporal_attention models on these downsampled corpora,
    saving the results to a csv given by `results_path`.

    This function was used to generate the results of experiment 5.3 in our paper.  
    The results_path file must be a .csv but it need not previously exist.
    If it does, results will be appended to it in the form: language, model_name, Spearman's rho.
    If it does not, it will be created.  The parent directory to the .csv file, however, 
    must exist before this function is used.
    
    Note that the following code can be used to map model names to author names, assuming
    that one wishes to use pandas to read in the results csv:
    ```
    results_path = "results/token_size_experiment.csv"

    df = pd.read_csv(results_path)
    df.replace("UG_Student_Intern", "Pömsl and Lyapin (2020)", inplace=True)
    df.replace("UWB", "Pražák et al. (2020)", inplace=True)
    df.replace("temporal_attention", "Rosin and Radinsky (2022)", inplace=True)
    ```
    """
    EXPS_PER_TKN_TRGT = 50

    # Add a new model here!
    fns_pack = {
        'UWB': evaluation_rules_UWB,
        'UG_Student_Intern': evaluation_rules_UGSI,
        'temporal_attention': evaluation_rules_TA
    }

    token_test_targets = range(250_000, 6_250_000+1, 500_000)
    with open(results_path, "a+") as results_file:
        writer = csv.writer(results_file, delimiter=",")

        # As in `run_stability_experiment`, have to account for situation where
        # this is the first experiment being run and situation where results already
        # exist and we don't want to overwrite them
        results_file.seek(0)
        if results_file.readline().strip("\n") != "Language,Model,Tokens,Spearman's Rho":
            writer.writerow(["Language", "Model", "Tokens", "Spearman's Rho"])

        for token_target in token_test_targets:
            # It could be interesting to run the token size experiments across
            # all languages
            result_packs = yield_results(corpora_path, token_target, EXPS_PER_TKN_TRGT, "english", fns_pack)
            for model_name, result in result_packs:
                writer.writerow(["english", model_name, token_target, result])
                results_file.flush()


def run_experiments():
    """Runs experiments 5.1, 5.2, and 5.3 in our paper."""
    corpora_path = "data/downsampled"
    results_dir = "results"
    Path(results_dir).mkdir(exist_ok=True, parents=True)

    run_stability_experiment(corpora_path, f"{results_dir}/stability_experiment.csv")
    run_token_size_experiment(corpora_path, f"{results_dir}/token_size_experiment.csv")

if __name__ == "__main__":
    run_experiments()
