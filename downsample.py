import argparse
import csv
import linecache
import os
import pathlib
import pickle
import random
import re
import shutil
import sys
import textwrap
from argparse import RawDescriptionHelpFormatter
from math import ceil
from tabulate import tabulate
from tqdm import tqdm

# not to be taken too seriously
class codes:
    SUCCESS = "\033[1m\033[32mSuccess:\033[0m"
    WARNING = "\033[1m\033[35mWarning:\033[0m"
    EXITING = "\033[1m\033[31mExiting:\033[0m"
    ERROR = "\033[1m\033[31mError:\033[0m"

    @staticmethod
    def color_name(name: str) -> str:
        return f"\033[36m{name}\033[0m"


#########################################
# Helpers                               #
#########################################

def clean(sentence: str) -> str:
    """Returns a sentence that has been cleaned of punctuation,
    OCR artifacts, duplicate and trailing spaces, nn and vb part of
    speech tags, and finally the letter x (more details on the last below).

    In order to cross reference an annotated use with its lemmatized counterpart,
    both sentence must be cleaned of small formatting differences that end up
    making a straightforward string comparison fail.  These formatting differences
    can be small—for instance the English annotated uses don't have pos tags,
    while in the ccoha datasets target words have been given simple pos tags
    as either _nn or _vb—or quite large.  The Swedish and German datasets, for
    instance, contain huge OCR artifacts (»~, ♦♦♦♦♦♦♦♦, !•m, /---*-, etc).

    The letter x must also be removed as there is inconsistent formatting in
    the German corpora when using x in the context of something like "2x4".
    Sometimes it is removed, sometimes it is not.  To avoid these cases,
    we just remove all x s from the input entirely.  

    There is also one other exception: in ccoha1.txt the line:

        "so after the famous christmas-dinner with its nice roast-meats and 
         pudding and pie after the game of romp with her father and the ride 
         on the rocking-horse with her brother who at last from mere mischief 
         have tip_vb her off and send her cry to her mother begin to think 
         about go there"
    
    corresponds to the following use of tip:

        "so , after the famous christmas-dinner with its nice roast-meats , 
         and pudding , and pie , - - after the game of romp with her father 
         , and the ride on the rocking-horse with her brother , who , at 
         last , from mere mischief , have tip her off , and send her cry to 
         her mother , ---she begin to think about go there ."
    
    The discerning documentation reader will notice that there is one word missing
    ("her mother begin to think" vs "her mother , ---she begin to think").  I
    think this is a weird error in the SemEval data, so this function also looks 
    out for a line that has the exact use given above and manually deletes "---she".  
    Note that of course we could just delete any occurences of "she" from all input
    sentences, but ultimately it is nice to change the sentences as little
    as possible so that sentences are more easily findable when error messages 
    are printed.

    >>> clean("its attack_nn -- be unwilling to be disturb")
    'its attack be unwilling to be disturb'

    >>> clean("its - - attack , -- be unwilling to be disturb !")
    'its attack be unwilling to be disturb'

    >>> clean("Immobilie rsn| «|an| wm| ftf Al im Fk| Tljft  ■  Achtung *&%  ")
    'Immobilie rsn an wm ftf Al im Fk Tljft Achtung'
    """
    manual_exception = (
        "so , after the famous christmas-dinner with its nice roast-meats ,"
        " and pudding , and pie , - - after the game of romp with her father"
        " , and the ride on the rocking-horse with her brother , who , at"
        " last , from mere mischief , have tip her off , and send her cry to"
        " her mother , ---she begin to think about go there ."
    )

    if sentence == manual_exception:
        sentence = re.sub("she", "", sentence)

    return re.sub("\s+", " ", re.sub(r"(_(?:nn|vb)|[^a-zA-Z0-9 ]|x)", "", sentence)).strip()


def sample_random_lines(path: str, lines: int, sample_size: int) -> list[str]:
    """Returns `sample_size` lines sampled at random from file located at `path`
    that has `lines` total lines.
    """
    random_idxs = random.sample(range(lines), sample_size)
    return [linecache.getline(path, i).strip() for i in random_idxs]


def get_targets(targets_path: str) -> list[str]:
    """Returns a list of target words in `targets_path` equivalent to the 
    words that were used as targets in the evaluation of SemEval 2020-Task 1.
    E.g., in English attack_nn, bag_nn, in German abbauen, abdecken, etc.

    Parameters
    ----------
    `targets_path` : str
        Should be path/to/targets.txt, where targets are seperated one on
        each line in targets.txt.  See program help string for information 
        on expected directory structure, but in short this should be 
        "data/semeval/{language}/targets.txt", where `language` in {"english",
        "german", "swedish"}.
    
    >>> get_targets("data/semeval/english/targets.txt")
    ['attack_nn', 'bag_nn', 'ball_nn', 'bit_nn', 'chairman_nn', 'circle_vb', ...]

    >>> get_targets("data/semeval/german/targets.txt")
    ['abbauen', 'abdecken', 'abgebrüht', 'Abgesang', 'Ackergerät', 'Armenhaus', ...]

    >>> get_targets("data/semeval/swedish/targets.txt")
    ['aktiv', 'annandag', 'antyda', 'bearbeta', 'bedömande', 'beredning', ...]
    """
    return [target.strip() for target in open(targets_path, "r")]


def read_uses_from_file(target_use_path: str, csv_column_of_use: int) -> list[str]:
    """Returns a list of clean annotated uses from `target_use_path`, where the column of
    the annotated use in each line of the csv file is given by `csv_column_of_use`.

    >>> read_uses_from_file("data/annotated_uses/english/tree_nn.csv", 12)
    ['53 represent a tree the low part of which have cease to grow in consequence of overfruiting', ...]
    """
    with open(target_use_path, "r") as uses_csv:
        reader = csv.reader(uses_csv, delimiter="\t", quotechar="`")
        next(reader) # skip first line

        return [clean(line[csv_column_of_use]) for line in reader]

def read_uses(target_use_paths: list[str]) -> list[str]:
    """Returns a list of clean annotated uses gathered from the csv file paths
    given in `target_use_paths`.

    Parameters
    ----------
    `target_use_paths` : list[str]
        Paths to csv files housing the annotated uses for a given target word in
        some language.  Should be a list of paths, where each individual path has
        the form "data/annotated_uses/{language}/{target}.csv", but see program 
        help for more info on directory structure.  Playing around with the csv files
        manually might help to get a sense for what their formatting is.  In general,
        however, the important part of each csv file is the lemmatized context column,
        which has the form of a space split string where each token has been lemmatized
        and is space seperated from other tokens.  For instance, in english "attack_nn":
        "commence the life of a pioneer , and that , in a night attack , his cabin have 
        be burn , his wife kill , and his son carry away by the savage ."
    
        E.g.,: ["data/annotated_uses/english/attack_nn.csv", "data/annotated_uses/english/bag_nn.csv"]

        Targets can be found from `get_targets`.

    Returns
    ----------
    list[str]
        List of annotated uses collected from csv files located in `target_use_paths`, where each 
        annotated use has been cleaned using the `clean` function.
    """
    uses = list()
    # if support for latin is added this line will have to change —
    # for latin the column is 6, not 12
    column_of_use_in_csv = 12 

    for target_use_path in target_use_paths:
        uses.extend(read_uses_from_file(target_use_path, column_of_use_in_csv))

    return uses

#########################################
# Verification                          #
#########################################

def _cross_verify_in_corpus(reference_map: dict[str, tuple[str, str]], corpus_path: str, corpus_len: int):
    """Populates `reference_map` dict with all uses found in `corpus_path`.

    Parameters
    ----------
    `reference_map` : dict[str, tuple[str, str]]
        A dictionary mapping cleaned annotated use sentences to tuples of the form
        (raw use in corpus, corpus name).  See Return info from `cross_verify` for
        more info on why this format is necessary.
    `corpus_path` : str
        Path to corpus to cross verify in, e.g., "data/semeval/english/ccoha1.txt"
    `corpus_lines` : int
        Total number of lines in corpus, for use when pretty printing a progress bar.
    """
    with open(corpus_path, "r") as corpus:
        for raw_sentence in tqdm(corpus, total=corpus_len):
            sentence = clean(raw_sentence)
            if sentence in reference_map:
                reference_map[sentence] = (raw_sentence.strip(), corpus_path)


def cross_verify(uses: list[str], corpus_paths: list[str], diagnostic_txt: str) -> list[tuple[str, str]]:
    """Cross references annotated `uses` with the corpora given in `corpus_paths`.

    This function searches through each corpus in `corpus_paths` and attempts to find a match
    for each use in `uses`.  A match is determined by using the `clean` function to first
    remove extraneous punctuation, OCR errors, etc, and then searching for an exact match.
    Diagnostic information is printed to the console based on the diagnostic_txt passed in,
    and if any `uses` were not found in the corpora then an error message is printed describing
    a few of the `uses` that weren't found and the program is exit.  While I included a small
    doctest demonstrating toy usage, to see how this function should be used in program flow it's 
    probably best to go to its call sites in `handle_downsample`.

    Parameters
    ----------
    `uses` : list[str]
        List of cleaned uses collected via `read_uses`.  
    `corpus_paths` : list[str]
        List of corpora to cross reference in, e.g.:
            ["data/semeval/english/ccoha1.txt", "data/semeval/english/ccoha2.txt"]
    `diagnostic_txt` : str
        This function will print diagnostic text followed by a space and the corpus path, so for instance 
        a caller could set diagnostic_txt="Searching for annotated uses from".
    
    Returns
    ----------
    list[tuple[str, str]]
        Tuples that have the following form:

            (cross referenced str found, corpus path where use was found)

        This may seem like a strange return format, but it's done for two principle reasons.  First,
        we want to preserve as much of the formatting from the SemEval datasets as possible.  When we
        downsample we don't want to introduce formatting changes that may cause unecessary noise.
        This desire to prevent noise is why we include the cross referenced string found.  Second, we
        will need to know where the cross reference was found when downsampling so have to include
        the full path as the second tuple element.

        **Note** that if any uses were not found this function will exit the program after printing
        a relevant error code.
    
    >>> uses = ["53 represent a tree the low part of which have cease to grow in consequence of overfruiting"]
    >>> paths = ["data/semeval/english/ccoha1.txt"]
    >>> cross_verify(uses, paths, "Searching for single use in") # doctest: +SKIP

    The following will be printed:

        Searching for single use in data/semeval/english/ccoha1.txt...

        100%|████████████████████████████████████████████████████████| 253644/253644 [00:01<00:00, 147203.85it/s]

    The following will be returned:

        [('53 represent a tree_nn the low part_nn of which have cease to grow in consequence of over-fruiting', 
            'data/semeval/english/ccoha1.txt')]
    """
    reference_map = {
        use: ("", "") for use in uses 
    }

    # "data/semeval/english/ccoha1.txt"
    # TODO: Document this
    langauge = next(lang for lang in ["english", "german", "swedish"] if lang in corpus_paths[0])
    cache_dir = f"data/cached_annotated_uses"
    cache_path = f"{cache_dir}/{langauge}.pkl"

    if os.path.isfile(cache_path):
        print(f"\nCached annotated uses found in {cache_path}")
        with open(cache_path, "rb") as cached_uses:
            return pickle.load(cached_uses)

    corpora_counts = [sum(1 for _ in open(corpus_path, "r")) for corpus_path in corpus_paths]

    for corpus_path, corpus_lines in zip(corpus_paths, corpora_counts):
        print(f"\n{diagnostic_txt} {corpus_path}...\n")
        _cross_verify_in_corpus(reference_map, corpus_path, corpus_lines)

    not_found = [use_sentence for use_sentence, (match, _) in reference_map.items() if not match]

    if not_found:
        print(f"\n{codes.ERROR} {len(not_found):,} uses could not be found, here were the beginning of the first few:\n")
        for missing_use in not_found[:5]:
            print(missing_use[:250])
            print()
        sys.exit(1)

    print(f"\nSaving cached uses to {cache_path}")

    pathlib.Path(cache_dir).mkdir(exist_ok=True, parents=True)
    with open(cache_path, "wb") as cache_file:
        pickle.dump(list(reference_map.values()), cache_file, protocol=pickle.HIGHEST_PROTOCOL)

    return reference_map.values()


#########################################
# Downsampling                          #
#########################################

def get_downsample_from_corpus(
    cross_references: list[tuple[str, str]],
    corpus_path: str, 
    target: int,
    disable_warn: bool = False
) -> set[str]:
    """Returns a set of downsampled lines from a corpus, where every cross referenced
    use in the corpus is included and a random sample of other lines are included
    until the `target` token count is reached.

    If there were more cross referenced uses in a corpus than requested tokens, a warning
    will be printed and the user will be asked to confirm that they wish to continue
    with the downsample.  If they do not select yes (y), then the program will be exit.
    Similarly, if the user asked for more tokens than exist in the target corpus an error
    message will be printed ("downsample" kind of necessitates making smaller, doesn't it?).
    
    If all goes well, a summary table will be printed with statistics highlighting
    the number of tokens sampled total, randomly, and from annotated uses.

    Parameters
    ----------
    `cross_references` : list[tuple[str, str]]
        Same format as the output of `cross_verify`. 
    `corpus_path` : str
        path/to/corpus.txt, e.g., "data/semeval/english/ccoha1.txt"
    `target` : int
        Token target.
    """

    corpus_lines = sum(1 for _ in open(corpus_path, "r"))

    # rough statistics to understand around how many lines to sample
    tokens_in_sentence = lambda sentence: len(sentence.split())

    # we should expect to have to sample around target / avg_tokens_per_line
    # number of lines to reach the desired token target... of course, we will have
    # to throw out some of the lines if we select any cross referenced ones, so we would prefer to 
    # select too many than too few (hence multiplication by 2) — tqdm is used because
    # this ends up being the longest wait in the downsampling process
    avg_tokens_per_line = sum(tokens_in_sentence(line) for line in tqdm(open(corpus_path), total=corpus_lines)) / corpus_lines
    num_lines_to_sample = min(corpus_lines, 2 * ceil(target / avg_tokens_per_line))

    downsample = {match for match, corpus in cross_references if corpus == corpus_path}
    total_uses_tokens = sum(tokens_in_sentence(sample) for sample in downsample)

    if total_uses_tokens >= target and not disable_warn:
        verification = input(
            f"\n{codes.WARNING} There were {total_uses_tokens:,} tokens in the annotated use"
            f" sentences.  This is more than the {target:,} requested tokens, but every annotated use"
            " must be included in the downsampled corpus, are you sure this is what you want? (y/n)\n\n >>> "
        )

        if verification.lower() != "y":
            print(f"\n{codes.EXITING} Note that if a downsample already occured it will not be reversed.")
            sys.exit(0)

    if corpus_lines * avg_tokens_per_line < target:
        print(
            f"\n{codes.ERROR} You asked for {target:,} tokens, but there are only"
            f" {round(corpus_lines * avg_tokens_per_line):,} total tokens in {corpus_path}."
            "  Maybe try again with fewer tokens?"
        )
        sys.exit(1)


    tokens_sampled = total_uses_tokens
    for sentence in sample_random_lines(corpus_path, corpus_lines, num_lines_to_sample):
        if tokens_sampled >= target:
            break
        if sentence in downsample:
            continue

        downsample.add(sentence)
        tokens_sampled += tokens_in_sentence(sentence)

    print("\n" + " "*3 + "Downsample Summary\n")
    print(tabulate(
        [["Annotated Uses", total_uses_tokens], ["Random Samples", tokens_sampled-total_uses_tokens],
         ["Total", tokens_sampled]], headers=["Category", "Tokens"]
    ))

    return downsample


def downsample(
    cross_references: list[tuple[str, str]],
    read_corpora_paths: list[str], 
    write_corpora_paths: list[str], 
    target: int,
    disable_warn: bool = False
):
    """Downsample corpora given in `read_corpora_paths` to .txt file paths
    given in `write_corpora paths` that have at least `target` total tokens and
    include every cross reference in `cross_references`.

    Every cross referenced instance is included in the downsampled text file,
    and it is assumed that `read_corpora_paths` and `write_copora_paths` are the
    same length and can be zipped together to give read / write paths.  So that
    read_corpora_paths[0] is the file that should be downsampled into write_copora_paths[0],
    and so on.  It is not the case that exactly `target` total tokens will be sampled,
    just based on the length of the last random sentence that was sampled before
    the token limit was reached, but generally at worse it will be + 50 or so additional
    tokens.  As described in the documentation for `get_downsample_from_corpus`, if
    there were more cross referenced tokens than requested tokens, the user will
    be prompted to confirm that they wish to continue with the downsample even though
    every annotated (i.e., cross referenced) use must be included to preserve ground
    truth data.  Similar error handling occurs if the user asks for too many tokens.

    Parameters
    ----------
    `cross_references` : list[tuple[str, str]]
        List of tuples that have the form: 
            (reference to include, corpus path of reference)
        where "reference to include" is some cross referenced annotated use 
        that must be included in the downsample, and "corpus path of reference" is
        the path to the corpus where the reference was taken from.  It is assumed 
        that "corpus path of reference" is one of the paths in `read_corpora_paths`.

        Really, this argument should just be the output of `cross_verify`.  
    `read_corpora_paths` : list[str]
        List of paths to the corpora that should be downsampled.
    `write_corpora_paths` : list[str]
        List of paths to the corpora to write to.  It is not necessary for these files
        to exist, but the parent directory structures must exist.  For example,
        if write_corpora_paths = ["data/downsampled/english/ccoha1.txt"], then the directory
        data/downsampled/english/ must exist, but it is not necessary that there is a file
        labelled ccoha1.txt in that directory.
    `target` : int
        Target number of tokens in the downsample.
    """
    for corpus_read_path, corpus_write_path in zip(read_corpora_paths, write_corpora_paths):
        print(f"\nDownsampling from {corpus_read_path} into {corpus_write_path}\n")
        downsampled_corpus = list(get_downsample_from_corpus(cross_references, corpus_read_path, target, disable_warn=disable_warn))
        random.shuffle(downsampled_corpus)
        with open(corpus_write_path, "w") as downsample_file:
            for sample in downsampled_corpus:
                downsample_file.write(f"{sample}\n")


def handle_downsample(language: str, target: int, write_path: str, disable_warn: bool = False):
    """Main driver that handles the downsample, printing
    relevant diagnostic text to console and querying user input
    when appropriate.  See help message for information on program flow.

    `language` must be in {"english", "german", "swedish"}.
    """
    langauge_to_corpus_names = {
        "english": ("ccoha1", "ccoha2"), 
        "german": ("dta", "bznd"), 
        "latin": ("latinISE1", "latinISE2"), 
        "swedish": ("kubhist2a", "kubhist2b")
    }

    names = langauge_to_corpus_names[language]
    corpus1_path = f"data/semeval/{language}/{names[0]}.txt"
    corpus2_path = f"data/semeval/{language}/{names[1]}.txt"
    corpus1_write_path = f"{write_path}/{language}/{names[0]}.txt"
    corpus2_write_path = f"{write_path}/{language}/{names[1]}.txt"
    targets_path = f"data/semeval/{language}/targets.txt"
    truth_path = f"data/semeval/{language}/truth"

    # makes the write directory in case it doesn't exist yet
    pathlib.Path(f"{write_path}/{language}").mkdir(parents=True, exist_ok=True)

    # copies over the truth and targets.txt files
    shutil.copyfile(targets_path, f"{write_path}/{language}/targets.txt")
    shutil.copytree(truth_path, f"{write_path}/{language}/truth", dirs_exist_ok=True)

    print("\n" + "="*25 + " Getting Target Words " + "="*25)
    print(f"\nGathering targets from data/semeval/{language}/targets.txt")

    targets = get_targets(targets_path)

    print("\n TARGETS:")
    print("-"*20)
    print("  " + "\n  ".join(textwrap.wrap(", ".join(targets), width=85)))

    target_use_paths = [f"data/annotated_uses/{language}/{target.lower()}.csv" for target in targets]
    
    read_paths = [corpus1_path, corpus2_path]
    write_paths = [corpus1_write_path, corpus2_write_path]

    uses = read_uses(target_use_paths)

    print("\n" + "="*25 + " Getting Target Uses " + "="*25)
    cross_references = cross_verify(uses, read_paths, "Searching for annotated uses from")

    print("\n" + "="*25 + " Downsampling " + "="*25)
    downsample(cross_references, read_paths, write_paths, target, disable_warn=disable_warn)

    print("\n" + "="*25 + " Verifying Target Uses in Downsampled " + "="*25)
    cross_verify(uses, write_paths, "Verifying annotated uses in")
    print()
    print(f"{codes.SUCCESS} Downsample completed and verified in {language.capitalize()}, see tables for summary statistics.")
    

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)

    language_help_txt = (
        "Which language to downsample from, supported languages are"
        " English, German, and Swedish."
    )

    target_help_txt = "The number of desired tokens after downsampling, defaults to 150k."
    seed_help_txt = "Random seed to use for setting random.seed(__), defaults to 42."
    write_help_txt = "Where to write the downsampled corpora to, defaults to data/downsampled."

    downsampler_description = textwrap.dedent("""
    A simple script for downsampling the SemEval 2020-Task 1 corpora while preserving
    manual annotated uses that were used to create ground truth data in SemEval.  
    Currently, downsampling is supported in English, German, and Swedish, but not in 
    Latin (I've not found a way to match Latin annotated uses to their lemmatized
    context in the SemEval dataset).  This script can be used to create corpora of any 
    target size below the total number of tokens per language corpora (e.g., there are 
    ~6.56 million tokens in ccoha1, you can't ask this program to "downsample" to 6.57 
    million tokens in English as there weren't even 6.57 million tokens to begin with!).  

    The script will write to "data/downsampled/{language}/" the downsampled corpora for the
    requested language by default, although that can be changed by passing in a different.  
    The downsampled corpora will have a randomized line order but will have the same lines run to 
    run as long as random_seed is not changed.  The truth/ and targets.txt files for the given
    language will also be copied over, so that after running this file the director format
    will be

        data/downsampled/{language}
            ├── truth
            │    ├── binary.txt
            │    └── graded.txt
            ├── CORPUS1.txt
            ├── CORPUS2.txt
            └── targets.txt

    where CORPUS1 and CORPUS2 are given by the name of the corpora being downsampled, and
    data/downsampled can be changed to the desired write directory using the -w flag.

    The program has four stages:

        1. Target words are searched for in the "SemEval 2020-Task 1 Targets" directory
           described below.
        2. Annotated uses from the "Annotated Uses" directory are cross referenced with the
           lemmatized corpora in the "SemEval 2020-Task 1 Corpora" directory.  This is done
           to ensure that we are not sampling uses twice and because the uses have a different
           format than their lemmatized counterpart in the SemEval corpora.  For instance, 
           the annotated uses do not have pos tags on target words, and there will be occasional
           formatting differences.  Here is one example:

            Annotated Use (for "tip"): "so , after the famous christmas-dinner with its 
                                        nice roast-meats , and pudding , and pie , - - 
                                        after the game of romp with her father , and the 
                                        ride on the rocking-horse with her brother , who 
                                        , at last , from mere mischief , have tip her off 
                                        , and send her cry to her mother , she begin to 
                                        think about go there ."
            
            SemEval Counterpart: "so after the famous christmas-dinner with its nice 
                                  roast-meats and pudding and pie after the game of romp 
                                  with her father and the ride on the rocking-horse with her 
                                  brother who at last from mere mischief have tip_vb her 
                                  off and send her cry to her mother she begin to think 
                                  about go there"
           
           We want to preserve as much of the SemEval properties as possible, so once each 
           annotated use is matched with its SemEval counterpart the SemEval version will be
           what is placed in the final downsampled corpus.
        3. Downsampling occurs, first including every annotated use and then randomly 
           sampling sentences until the target token count is reached.  At this stage
           summary statistics describing the downsample will be printed in the following
           format:

                   Downsample Summary

                Category          Tokens
                --------------  --------
                Annotated Uses         #
                Random Samples         #
                Total                  #
            
           Where "Annotated Uses" corresponds to the number of tokens that were in the
           annotated use sentences and had to be included, "Random Samples" corresponds
           to the number of tokens from sentences that were randomly sampled until the
           target token amount was reached, and "Total" corresponds to the total number
           of tokens sampled.  If the number of requested tokens is fewer than the 
           number of tokens in annotated uses a warning is raised and you will be asked
           to confirm that you wish to proceed with the downsampling process even though
           there will be more tokens than requested (remember that every annotated use
           must be included in the downsampled corpora).
        4. Finally, a verification step ensures that every annotated use is actually 
           present in the downsampled corpora.

    This program expect a directory structure in the following form:

        1. Annotated Uses: data/annotated_uses/{language}/{target_word}.csv
        2. SemEval 2020-Task 1 Corpora: data/semeval/{language}/{corpus_name}.txt
        3. SemEval 2020-Task 1 Targets: data/semeval/{language}/targets.txt

    Where:

        1. "Annotated uses" are csv files corresponding to the manually annotated 
           uses from "DWUG: A large Resource of Diachronic Word use Graphs in Four Languages,"
           a 2021 paper by Schlechtweg et al.  These uses were what were used to create the
           ground truth data for SemEval 2020-Task 1.
        2. "SemEval 2020-Task 1 Corpora" are the .txt corpora corresponding to a given language
           used in SemEval-Task 1.  These corpora can be found from the 2020 paper by 
           Schlechtweg et al, "SemEval-2020 Task 1: Unsupervised Lexical Semantic Change Detection"
        3. "SemEval 2020-Task 1 Targets" ibid except the .txt target files

    This can be created by running `./download.sh` (you may need to give download.sh
    executable privileges first).

    Example uses (output elided, for full output example see README.md):

    >>> python downsample.py english
    (details elided)

    >>> python downsample.py german
    (details elided, but this one will ask you to confirm that you want to continue
     the downsample despite the fact that the number of tokens requested is fewer
     than the number of annotated use tokens)

    >>> python downsample.py english -t 165000 -s 41
    (details elided, but essentially the same as the first example just using a 
     different seed and asking for more tokens to be sampled)

    """)

    parser = argparse.ArgumentParser(description=downsampler_description, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('language', help=language_help_txt)
    parser.add_argument('-t', '--target_tokens', type=int, help=target_help_txt, default=150000)
    parser.add_argument('-s', '--random_seed', type=int, help=seed_help_txt, default=42)
    parser.add_argument('-w', '--write_path', type=str, help=write_help_txt, default="data/downsampled")
    args = parser.parse_args()

    LANGUAGE = args.language.lower()
    random.seed(args.random_seed)

    if LANGUAGE in {"english", "german", "swedish"}:
        handle_downsample(LANGUAGE, args.target_tokens, args.write_path)
    else:
        print(f"{codes.ERROR} {LANGUAGE} is not supported.  Only English, German, and Swedish can be downsampled.")
