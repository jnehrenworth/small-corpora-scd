import argparse
import csv
import linecache
import random
import re
import textwrap
import sys
from math import ceil
from tabulate import tabulate
from tqdm import tqdm
from typing import TextIO

# not to be taken too seriously
class codes:
    SUCCESS = "\033[1m\033[36mSuccess:\033[0m"
    WARNING = "\033[1m\033[35mWarning:\033[0m"
    EXITING = "\033[1m\033[31mExiting:\033[0m"
    ERROR = "\033[1m\033[31mError:\033[0m"


#########################################
# Helpers                               #
#########################################

def clean(sentence: str) -> str:
    # includes punctuation but also ocr artifacts — actually ends up being a total
    # hassle... 
    # x for things like 2x4
    # double replace for things like "the jester from , leicster"
    # strip b/c of annoying final things like "the jester from , leicster !"
    return re.sub("\s+", " ", re.sub(r"(_(?:nn|vb)|[^a-zA-Z0-9 ]|x)", "", sentence)).strip()


def sample_random_lines(file_name: str, file_lines: int, num_desired_lines: int) -> list[str]:
    random_idxs = random.sample(range(file_lines), num_desired_lines)
    return [linecache.getline(file_name, i).strip() for i in random_idxs]


def get_target_use_path(target: str, language: str) -> str:
    return f"data/annotated_uses/{language}/{target}.csv"

def get_targets(targets_path: str) -> list[str]:
    return [target.strip() for target in open(targets_path, "r")]


def read_uses_from_file(uses_csv_file: TextIO, csv_column_of_use: int) -> list[str]:
    reader = csv.reader(uses_csv_file, delimiter="\t", quotechar="`")
    next(reader) # skip first line

    return [clean(line[csv_column_of_use]) for line in reader]

def read_uses(target_use_paths: list[str], csv_column_of_use: int) -> list[str]:
    uses = list()

    for target_use_path in target_use_paths:
        # en/de/sv: context_lemmatized; la:indexes_target_sentence
        with open(target_use_path, "r") as use_file:
            uses.extend(read_uses_from_file(use_file, csv_column_of_use))

    return uses

#########################################
# Verification                          #
#########################################

def cross_verify_in_corpus(verified_uses: dict[str, tuple[str, str]], corpus: TextIO, corpus_name: str, corpus_len: int):
    for raw_sentence in tqdm(corpus, total=corpus_len):
        sentence = clean(raw_sentence.strip())
        if sentence in verified_uses:
            verified_uses[sentence] = (raw_sentence.strip(), corpus_name)


def cross_verify(uses: list[str], corpus_paths: list[str], corpus_names: list[str], diagnostic_txt: str) -> dict[str, str]:
    # tuple: (use, corpus_name)
    verified_uses = {
        use: ("", "") for use in uses 
    }

    corpora_counts = [sum(1 for _ in open(corpus_path, "r")) for corpus_path in corpus_paths]

    for corpus_path, corpus_name, corpus_lines in zip(corpus_paths, corpus_names, corpora_counts):
        with open(corpus_path, "r") as corpus:
            print(f"\n{diagnostic_txt} {corpus_path}...\n")
            cross_verify_in_corpus(verified_uses, corpus, corpus_name, corpus_lines)

    not_found = [use_sentence for use_sentence, (match, _) in verified_uses.items() if not match]

    if not_found:
        print(f"\n{codes.ERROR} {len(not_found):,} uses could not be found, here were the beginning of the first five:\n")
        for missing_use in not_found[:5]:
            print(missing_use[:250])
            print()
        sys.exit(1)

    return verified_uses

#########################################
# Downsampling                          #
#########################################

def get_downsample_from_corpus(
    verfied_uses: dict[str, tuple[str, str]], 
    corpus_path: str, 
    corpus_name: str, 
    target: int
) -> set[str]: # returning a set has the added benefit of randomizing...
    corpus_lines = sum(1 for _ in open(corpus_path, "r"))

    # the average number of tokens per sentence is ~25.86 in ccoha1 and 
    # ~19.14 in ccoha2 — we should expect then to have to sample around target / avg_tokens_per_line
    # number of lines to reach the desired token target... of course, we will have
    # to throw out some of the lines if we select any verified ones, so we would prefer to 
    # select too many than too few (hence multiplication by 2)
    tokens_in_sentence = lambda sentence: len(sentence.split())

    # tqdm here because this ends up being the longest wait
    avg_tokens_per_line = sum(tokens_in_sentence(line) for line in tqdm(open(corpus_path), total=corpus_lines)) / corpus_lines
    num_lines_to_sample = min(corpus_lines, 2 * ceil(target / avg_tokens_per_line))

    downsample = {match for match, corpus in verfied_uses.values() if corpus == corpus_name}
    total_uses_tokens = sum(tokens_in_sentence(sample) for sample in downsample)

    if total_uses_tokens >= target:
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
    verified_uses: dict[str, tuple[str, str]], 
    read_corpora: list[str], 
    write_corpora: list[str], 
    corpus_names: list[str],
    target: int
):
    for corpus_read_path, corpus_write_path, corpus_name in zip(read_corpora, write_corpora, corpus_names):
        print(f"\nDownsampling from {corpus_read_path} into {corpus_write_path}\n")
        downsampled_corpus = get_downsample_from_corpus(verified_uses, corpus_read_path, corpus_name, target)
        with open(corpus_write_path, "w") as downsample_file:
            for sample in downsampled_corpus:
                downsample_file.write(f"{sample}\n")


def handle_downsample(language: str, target: int):
    language_to_use_col = {"english": 12, "german": 12, "latin": 6, "swedish": 12}
    langauge_to_corpus_names = {
        "english": ("ccoha1", "ccoha2"), 
        "german": ("dta", "bznd"), 
        "latin": ("latinISE1", "latinISE2"), 
        "swedish": ("kubhist2a", "kubhist2b")
    }

    names = langauge_to_corpus_names[language]
    corpus1_path = f"data/corpora/semeval/{language}/{names[0]}.txt"
    corpus2_path = f"data/corpora/semeval/{language}/{names[1]}.txt"
    corpus1_write_path = f"data/corpora/downsampled/{language}/{names[0]}.txt"
    corpus2_write_path = f"data/corpora/downsampled/{language}/{names[1]}.txt"

    print("\n" + "="*25 + " Getting Target Words " + "="*25)

    print(f"\nGathering targets from data/corpora/semeval/{language}/targets.txt")

    targets = get_targets(f"data/corpora/semeval/{language}/targets.txt")

    print("\n TARGETS:")
    print("-"*20)
    print("  " + "\n  ".join(textwrap.wrap(", ".join(targets), width=85)))

    target_use_paths = [get_target_use_path(target, f"{language}") for target in targets]
    
    read_paths = [corpus1_path, corpus2_path]
    write_paths = [corpus1_write_path, corpus2_write_path]
    names = [f"{names[0]}", f"{names[1]}"]

    uses = read_uses(target_use_paths, language_to_use_col[language])

    print("\n" + "="*25 + " Getting Target Uses " + "="*25)
    verified_uses = cross_verify(uses, read_paths, names, "Searching for annotated uses from")

    print("\n" + "="*25 + " Downsampling " + "="*25)
    downsample(verified_uses, read_paths, write_paths, names, target)

    print("\n" + "="*25 + " Verifying Target Uses in Downsampled " + "="*25)
    cross_verify(uses, write_paths, names, "Verifying annotated uses in")
    print()
    print(f"{codes.SUCCESS} Downsample completed and verified in {language.capitalize()}, see tables for summary statistics.")
    

if __name__ == "__main__":
    from argparse import RawDescriptionHelpFormatter

    language_help_txt = (
        "Which language to downsample from, supported languages are"
        " English, German, and Swedish."
    )

    target_help_txt = "The number of desired tokens after downsampling, defaults to 150k."
    seed_help_txt = "Random seed to use for setting random.seed(__), defaults to 42."

    downsampler_description = textwrap.dedent("""
    A simple script for downsampling the SemEval 2020-Task 1 corpora while preserving
    manual annotated uses that were used to create ground truth data in SemEval.  
    Currently, downsampling is supported in English, German, and Swedish, but not in 
    Latin as I've not found a way to match Latin annotated uses to their lemmatized
    context in the SemEval dataset.  This script can be used to create corpora of any 
    target size below the total number of tokens per language corpora (e.g., there are 
    ~6.56 million tokens in ccoha1, you can't ask this program to "downsample" to 6.57 
    million tokens in English as there weren't even 6.57 million tokens to begin with!).  

    The script will write to "data/downsampled/{language}/" the downsampled corpora for the
    requested language.  The downsampled corpora will have a randomized line order but will 
    have the same lines run to run as long as random_seed is not changed.

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
        2. SemEval 2020-Task 1 Corpora: data/corpora/semeval/{language}/{corpus_name}.txt
        3. SemEval 2020-Task 1 Targets: data/corpora/semeval/{language}/targets.txt
        4. Downsample Write Directories: data/downsampled/{language}/

    Where:

        1. "Annotated uses" are csv files corresponding to the manually annotated 
           uses from "DWUG: A large Resource of Diachronic Word use Graphs in Four Languages,"
           a 2021 paper by Schlechtweg et al.  These uses were what were used to create the
           ground truth data for SemEval 2020-Task 1.
        2. "SemEval 2020-Task 1 Corpora" are the .txt corpora corresponding to a given language
           used in SemEval-Task 1.  These corpora can be found from the 2020 paper by 
           Schlechtweg et al, "SemEval-2020 Task 1: Unsupervised Lexical Semantic Change Detection"
        3. "SemEval 2020-Task 1 Targets" ibid except the .txt target files
        4. Where to write the downsampled corpora, note that it is expected that these directories
           already exist.

    But you haven't been messing with the directory structure, have you? ;)


    Example uses:

    >>> python3 downsample.py english
    (details elided)

    >>> python3 downsample.py german
    (details elided, but this one will ask you to confirm that you want to continue
     the downsample despite the fact that the number of tokens requested is fewer
     than the number of annotated use tokens)

    >>> python3 downsample.py english 165000 41
    (details elided, but essentially the same as the first example just using a 
     different seed and asking for more tokens to be sampled)

    """)

    parser = argparse.ArgumentParser(description=downsampler_description, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('language', help=language_help_txt)
    parser.add_argument('target_tokens', nargs='?', type=int, help=target_help_txt, default=150000)
    parser.add_argument('random_seed', nargs='?', type=int, help=seed_help_txt, default=42)
    args = parser.parse_args()

    LANGUAGE = args.language.lower()
    random.seed(args.random_seed)

    if LANGUAGE in {"english", "german", "swedish"}:
        handle_downsample(LANGUAGE, args.target_tokens)
    else:
        print(f"{codes.ERROR} {LANGUAGE} is not supported.  Only English, German, and Swedish can be downsampled.")
    
