"""Main evaluation script, run python3 evaluate.py -h for usage information.

Broadly, this script works by first populating the dataset paths expected
by each model with the `read_path` passed in (or "data/downsampled" by default), 
training each model on the populated datasets, and then evaluating each
model's performance.  The two most important functions in the codebase are
`populate` and `evaluate`.  The `populate` function takes a function that
describes rules for how to populate the datapaths for some model.  The `evaluate`
function works similarly except it expects a function describing how to evaluate
a certain model.  To add a new model, then, one just has to write functions 
with rules for how to populate the model's datapaths and how to evaluate the
model's performance.  See doc strings for `populate` and `evaluate` for 
more info.

The rest of this codebase is divided into sections with rules for each model
and then at the end the `populate` and `evaluate` functions and some silly
pretty printing.
"""
import argparse
import contextlib
import os
import pathlib
import shutil
import textwrap
from argparse import RawTextHelpFormatter
from tabulate import tabulate
from typing import Callable, Optional

# UG_Student_Intern fns
from models.UG_Student_Intern.predict import predict
from models.UG_Student_Intern.evaluate import evaluate_experiment

# UWB fns
import models.UWB.cca.embeddings.embeddings_generator as UWB_train
import models.UWB.cca.compare as UWB_eval

# just for nice printing
from downsample import codes


#########################################
# UG_Student_Intern Rules               #
#########################################

def path_rules_UGSI(language: str, file: str) -> Optional[str]:
    """Returns either a string representing the path the UG_Student_Intern model
    expects data for the given file in the given language to be, or None if
    the model does not expect or care about the given file (e.g., their model
    does not need to `target.txt` file to be copied over in any language).

    The UG_Student_Intern team has really good documentation, so the places they
    expect data to be written are fairly easy to find.  They provide a script,
    `download_semeval_data.sh`, that downloads the semeval dataset and places 
    its corpora into relevant directories.  One can see, from that script, where
    the UG_Student_Intern model expects data to live.  This function does not
    create any additional directories, as the UG_Student_Intern team created
    corpora-less stubs that essentially just need to be filled with the relevant
    corpora.

    If you haven't looked at `populate` yet, it's the best place to go to understand
    how this kind of function is expected to work.

    >>> path_rules_UGSI("english", "ccoha1.txt")
    'models/UG_Student_Intern/datasets/en-semeval/c1.txt'

    >>> path_rules_UGSI("swedish", "kubhist2b.txt")
    'models/UG_Student_Intern/datasets/sw-semeval/c2.txt'

    >>> path_rules_UGSI("swedish", "targets.txt") is None
    True
    """
    base_path = "models/UG_Student_Intern/datasets"
    acronyms_UG = {
        "english": "en",
        "german": "de",
        "swedish": "sw"
    }

    corpus_name_to_UG_name = {
        "ccoha1.txt": "c1.txt",
        "ccoha2.txt": "c2.txt",
        "dta.txt": "c1.txt",
        "bznd.txt": "c2.txt",
        "kubhist2a.txt": "c1.txt",
        "kubhist2b.txt": "c2.txt"
    }

    if file in corpus_name_to_UG_name:
        acronym = acronyms_UG[language]
        name = corpus_name_to_UG_name[file]

        populate_path = f"{base_path}/{acronym}-semeval/{name}"
        return populate_path
    
    return None


def evaluation_rules_UGSI(language: str) -> float:
    """Trains the UG_Student_Intern model on a given language and evaluates its 
    performance, returning Spearman's rho measuring the strength of correlation
    between the models prediction of ranked sentiment and ground truth rankings.

    This function expects that the UG_Student_Intern model has already had its
    datasets populated, perhaps by the `populate` function.  See `evaluate` for
    more information on how evaluation functions are expected to work.  
    """
    language_to_short = {
        "english": "en",
        "german": "de",
        "swedish": "sw"
    }

    short_lang = language_to_short[language]
    dataset_dir = f"models/UG_Student_Intern/datasets/{short_lang}-semeval/"
    output_dir = f"models/UG_Student_Intern/experiments/context-free_{short_lang}-semeval/"

    # it's quite easy to train and evaluate the UG_Student_Intern model
    # the predict function is responsible for the training, and the 
    # evaluate function does the (as one would expect) evaluation
    # 
    # see their documentation for more details, they actually have a pretty
    # understandable codebase
    predict(model_name="context-free", dataset_dir=dataset_dir, overwrite=True)
    print(f"\n{codes.color_name('UG_Student_Intern')} model evaluated in {language}")

    # evaluate_experiment returns (rho, p_val)
    return evaluate_experiment(dataset_dir, output_dir)[0]


#########################################
# UWB Rules                             #
#########################################   

def path_rules_UWB(language: str, file: str) -> Optional[str]:
    """Returns either a string representing the path the UG_Student_Intern model
    expects data for the given file in the given language to be, or None if
    the model does not expect or care about the given file.

    The UWB team did not create directory stubs for each corpora, so this
    function will also create any non-existent directories so that the returned
    path/to/file can be opened or copied to without having to do any directory
    creation.  To see whether the UWB model expects datasets to live, the best
    place to look is their config.py file in the UWB directory.  You have to
    do a little bit of detective-reverse engineering, so be prepared for that.

    >>> path_rules_UWB("english", "ccoha1.txt")
    'models/UWB/cca/data/test-data/english/semeval2020_ulscd_eng/corpus1/lemma/ccoha1.txt'
    
    >>> path_rules_UWB("swedish", "kubhist2b.txt")
    'models/UWB/cca/data/test-data/swedish/semeval2020_ulscd_swe/corpus2/lemma/kubhist2b.txt'

    >>> path_rules_UWB("swedish", "targets.txt")
    'models/UWB/cca/data/test-data/swedish/semeval2020_ulscd_swe/targets.txt'
    """
    base_path = "models/UWB/cca/data/test-data"
    corpus_name_to_corpus = {
        "ccoha1.txt": "corpus1",
        "ccoha2.txt": "corpus2",
        "dta.txt": "corpus1",
        "bznd.txt": "corpus2",
        "kubhist2a.txt": "corpus1",
        "kubhist2b.txt": "corpus2"
    }

    acronym = language[:3]
    dir_path = f"{base_path}/{language}/semeval2020_ulscd_{acronym}"

    # make directory if it doesn't previously exist
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)

    if file.endswith(".txt"):
        if file != "targets.txt":
            corpus_path = f"{dir_path}/{corpus_name_to_corpus[file]}/lemma"

            # again, creates dir if it hasn't already been created
            pathlib.Path(corpus_path).mkdir(parents=True, exist_ok=True)

            populate_path = f"{corpus_path}/{file}"
        else:
            populate_path = f"{dir_path}/{file}"

        return populate_path
        
    return None


def evaluation_rules_UWB(language: str) -> float:
    """Trains the UWB model on a given language and evaluates its performance, 
    returning Spearman's rho measuring the strength of correlation between the 
    models prediction of ranked sentiment and ground truth rankings.

    This function expects that the UWB model has already had its datasets populated, 
    perhaps by the `populate` function. 
    """
    language_to_train_eval = {
        "english": (UWB_train.train_english, UWB_eval.run_english_default),
        "german": (UWB_train.train_german, UWB_eval.run_german_default),
        "swedish": (UWB_train.train_swedish, UWB_eval.run_swedish_default)
    }
    
    # it seems as if there might be a bug in the UWB codebase in that the model expects
    # a tmp directory to already have been created. paradoxically, calling
    # `UWB_eval.delete_tmp_dir()` creates the necessary tmp directory 
    UWB_eval.delete_tmp_dir()

    # much of the rest of this code was pieced together by trying to follow
    # the `main` method of the compare.py file in the UWB directory
    # 
    # it's a dangerous place in there
    # 
    # `eval_pack` has the form:
    #      task_1_dir, task_2_dir, reverse_emb, use_bin_thld, use_nearest_neigbh,
    #      emb_type, emb_dim, window, epochs, mean_centering, unit_vectors
    # *don't change these parameters*: in order to change any parameters
    # one must first manually change the training parameters in embeddings_generator.py
    # in the UWB/cca/embeddings directory and only then change these to match them
    # also note that in their codebase epochs is written as iter, but see:
    # https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4
    task_1_dir, task_2_dir, _, _ = UWB_eval.init_folders('post-test')
    eval_pack = (
        task_1_dir, task_2_dir, False, True, False, 
        "w2v", 100, 5, 5, False, False
    )

    train, _eval = language_to_train_eval[language]

    train()

    # their eval method is incredibly noisy, this gives it a muzzle
    # _eval returns (p_val, rho)
    with contextlib.redirect_stdout(None):
        _, rho = _eval(*eval_pack)
        
    print(f"\n{codes.color_name('UWB')} model evaluated in {language}")
    
    return rho


def populate(path_rules: Callable[[str, str], Optional[str]], read_path: str):
    """Populates the datapaths for a given model with the corpora and data from 
    `read_path` following the rules specified in `path_rules`.

    The general idea is that each model evaluated can be trained by placing data in 
    some directory (following a certain directory structure) that a given model expects.
    If we understand the rules of how the directory structure was created we can recreate
    it programmatically by copying data in each language director from `read_path` according
    to the rules given by `path_rules`.  See `read_path` parameter info for expected
    directory structure of `read_path` and `path_rules` parameter info for how to go about
    creating a new `path_rules` fn.  After calling this function, it should be safe to 
    train the corresponding model to `path_rules`.

    Parameters
    ----------
    `path_rules` : Callable[[str, str], Optional[str]]
        The `path_rules` functions should take two inputs, language and a file name
        in the form `path_rules(language, file)`.  The variable `language` will be in
        {"english", "german", "swedish"} (in lowercase), and file will be in
        {"truth", "targets.txt", "ccoha1.txt", "ccoha2.txt", "dta.txt", "bznd.txt", 
        "kubhist2a.txt", "kubhist2b.txt"}.  (If it is helpful, `path_rules` may assume
        that the corpora names only appear as `file` when called with the relevant language,
        e.g., "ccoha1.txt" will only appear when `language` is "english").
        
        From these two intputs, `path_rules` should return either None to signal that
        it is not necessary to copy the given file, or a path where the file should be
        copied to.  The path should start from the current directory for this file or be 
        an absolute path.  For instance, the UG_Student_Intern model expects "ccoha1.txt" 
        to be placed in the path "models/UG_Student_Intern/datasets/en-semeval/c1.txt",
        so if the `path_rules` function was passed in to be `path_rules_UGSI`, then
        `path_rules("english", "ccoha1.txt")` => "models/UG_Student_Intern/datasets/
        en-semeval/c1.txt" (or the equivalent absolute version).
        
        It is expected that after calling `path_rules`, if a file path is returned then
        all parent directories exist on the way to that file.  See `path_rules_UWB` for
        an example of why this may be necessary and how to do this.

    `read_path` : str
        The `read_path` string should be a string with a path starting from this files
        working directory pointing to a directory which should be used to populate model
        datasets.  It is expected that the directory pointed to has the following structure:

            read_path
                ├── english
                │    ├── truth
                │    │    ├── binary.txt
                │    │    └── graded.txt
                │    ├── ccoha1.txt
                │    ├── ccoha2.txt
                │    └── targets.txt
                ├── german
                │    ├── truth
                │    │    ├── binary.txt
                │    │    └── graded.txt
                │    ├── dta.txt
                │    ├── bznd.txt
                │    └── targets.txt
                └── swedish
                     ├── truth
                     │    ├── binary.txt
                     │    └── graded.txt
                     ├── ccoha1.txt
                     ├── ccoha2.txt
                     └── targets.txt
            
            A common workflow might look like using `downsample.py` to downsample the 
            english, german, and swedish corpora to some target size and then this
            function can be used to populate model directories with the new downsized
            copora.  The only other files that should be in the read_path directory
            are those that are hidden (i.e., that start with a . — e.g., .DS_Store, .git).  If that becomes a hassle, it should be easy enough to change this function to
            ignore other files as well.
    """
    for language in os.listdir(f"{read_path}"):
        # ignore .DS_Store, etc.
        # change this line if you want to ignore other files in addition to hidden . files
        if language.startswith("."):
            continue
            
        for file in os.listdir(f"{read_path}/{language}"):
            cpy_path = f"{read_path}/{language}/{file}"
            populate_path = path_rules(language, file)
            if populate_path is not None:
                print(f"Writing {cpy_path} to {populate_path}")
                shutil.copyfile(cpy_path, populate_path)


def evaluate(evaluation_rules: Callable[[str], float]) -> dict[str, float]:
    """Returns a dict mapping language to Spearman's rho calculated via
    the `evaluation_rules` fn.

    Note that it is assumed that all relevant datapaths have been populated
    prior to this function being called (perhaps by the `populate` function).

    Parameters
    ----------
    `evaluation_rules` : Callable[[str], float]
        This function should take a language in {"english", "german", "swedish"}
        (can assume lowercase) and return a float representing Spearman's rho
        calculated by training the model on that language and then comparing
        the model's predicted performance against the ground truth ranking.  
        See `evaluation_rules_UGSI` for an example of how this can be done.

        The `evaluation_rules` function can expect that all relevant datapaths
        have previously been populated, so that the only role of the `evaluation_rules`
        function should be to train the model on the corpora of a given language,
        predict ranked sentiment, and then return Spearman's rho comparing
        predicted to ground truth rankings.

        It can be nice if the `evaluation_rules` function has some pretty printing
        that keeps the user appraised of training progress, but of course that is
        not necessary.

    Returns
    -------
    dict[str, float]
        A dictionary mapping languages to Spearman's rho calculated via the `evaluation_rules`
        fn.  Currently, only "english", "german", and "swedish" are evaluated, so the 
        dict format will be:

        {
            "english": x.xx,
            "german": y.yy,
            "swedish": z.zz
        }
    """
    results = dict()

    for language in ["english", "german", "swedish"]:
        print() # just for prettier spacing 
        results[language] = evaluation_rules(language)

    return results


def populate_all(read_path: str):
    """Calls the `populate` function on each model currently under testing and
    pretty prints some output to help the user understand what is going on.  After
    this function has been called, all models will have had their datapaths filled
    and the `evaluate_all` function can be called.
    """
    model_to_rules = {
        'UWB': path_rules_UWB,
        'UG_Student_Intern': path_rules_UGSI
    }

    underscore_len = lambda model_name: len(f"Populating for {model_name}")

    for model, rule in model_to_rules.items():
        print(f"Populating for {codes.color_name(model)}")
        print("-"*underscore_len(model) + "\n")
        populate(rule, read_path)
        print(f"\n{codes.SUCCESS} {codes.color_name(model)} datasets populated\n")


def evaluate_all():
    """Calls the `evaluate` function on each model currently under testing, pretty
    printing progress and a nice table of Spearman's rank correlation coefficients
    for each model in each language once the evaluation is completed.  Each model
    *must* have had their datapaths populated prior to this function being called
    (perhaps by the `populate_all` function ;) )
    """
    model_to_eval_rules = {
        'UWB': evaluation_rules_UWB,
        'UG_Student_Intern': evaluation_rules_UGSI
    }

    evaluations = list()
    underscore_len = lambda model_name: len(f"Evaluating {model_name}")

    for model, eval_rule in model_to_eval_rules.items():
        print(f"Evaluating {codes.color_name(model)}")
        print("-"*underscore_len(model))

        evaluations.append([codes.color_name(model)])
        evaluations[-1].extend(evaluate(eval_rule).values())

        print(f"\n{codes.SUCCESS} {codes.color_name(model)} model evaluated on all corpora\n")
    
    print("="*25 + " Evaluation Summary " + "="*25 + "\n")

    print("Reporting Spearman's rank correlation coefficient for each model in each language...\n")

    print(tabulate(evaluations, headers=["Model", "English", "German", "Swedish"]))
    print()


if __name__ == "__main__":
    import doctest
    doctest.testmod()

    rp_help = textwrap.dedent("""\
        Path to the directory housing corpora and data to use when evaluating models.  This
        directory must have the exact form:

            read_path
            ├── english
            │    ├── truth
            │    │    ├── binary.txt
            │    │    └── graded.txt
            │    ├── ccoha1.txt
            │    ├── ccoha2.txt
            │    └── targets.txt
            ├── german
            │    ├── truth
            │    │    ├── binary.txt
            │    │    └── graded.txt
            │    ├── dta.txt
            │    ├── bznd.txt
            │    └── targets.txt
            └── swedish
                 ├── truth
                 │    ├── binary.txt
                 │    └── graded.txt
                 ├── ccoha1.txt
                 ├── ccoha2.txt
                 └── targets.txt
                
        The only other files that can be in the read_path directory or any subdirectories
        are those that start with a dot and are hidden (e.g., .DS_Store, .git).  The
        read_path path should start from the current files working directory or else be
        and absolute path.  Defaults to data/downsampled if a read path is not provided.
    """)

    evaluate_description = textwrap.dedent("""
        This script creates summary statistics evaluating the predictive performance
        of each model in models/ on Subtask 2 of SemEval 2020-Task 1.  In particular,
        for each language—currently the supported languages are English, German, and
        Swedish—each model is used to create a predicted ranking of the target words
        given in read_path/{language}/targets.txt by amount of semantic change. Next, 
        the predicted ranking is compared to the ground truth ranking given in 
        read_path/{language}/truth/graded.txt and evaluated using Spearman's
        rank correlation coefficient.  

        After each model has been evaluated in each language, a table with
        summary results describing each model's performance in each language is
        printed.  To validate the results of my small-corpora testing, simply
        run `python3 evaluate.py`.  If you wish to create your own downsampled
        dataset, first use `download.sh` to download the SemEval 2020-Task 1 corpora
        and then use `downsample.py` to downsample the corpora to your hearts content.
        After a downsampled corpora has been created, run `python3 evaluate.py` in 
        order to evaluate the performance of models on the downsampled corpora.  If
        you do not wish to overwrite the existing downsample that I have created, 
        provide a write directory to `downsample.py` and pass that same directory
        to this program.  See README.md for example use.
    """)

    parser = argparse.ArgumentParser(description=evaluate_description, formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        'read_path', nargs='?', help=rp_help, type=str, default="data/downsampled"
    )

    args = parser.parse_args()

    read_path = args.read_path

    print("\n" + "="*25 + " Populating Dataset Paths " + "="*25 + "\n")
    populate_all(read_path)

    print("="*25 + " Evaluating Models " + "="*25 + "\n")
    evaluate_all()
    