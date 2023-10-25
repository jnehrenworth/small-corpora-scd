# small-corpora-scd
This repository accompanies our forthcoming paper, "Literary Intertextual Semantic Change Detection: Application and Motivation for Evaluating Models on Small Corpora" (permanent link available soon), appearing in the 4th International Workshop on Computational Approaches to Historical Language Change 2023 (LChange'23).  Here, you can find corpora, models, and testing scripts for the evaluation of semantic change detection (scd) models on small corpora.  See our paper for a general overview of the tasks and evaluation results, or read on for a repository primer.  If you have any questions, please don't hesitate to reach out (point of contact: Jackson Ehrenworth; jacksonehrenworth[@]gmail[.]com).  We welcome pull requests or other contributions.

## Table of Contents

- [Repository Tour](#repository-tour) 
- [Install](#install)
- [Usage](#usage)
- [Important Caveats](#important-caveats)
- [Citation](#citation)

## Repository Tour

* `data/` houses downsampled, downloaded, and cached datasets.
* `models/` houses all models under evaluation.  Incidentally, if you wish to add your own model this would be the place to do it.  See the documentation attached to `evaluate.py` to get a sense for how one can do this.
* `results/` houses the raw data informing the results presented in our paper.
* `downsample.py` can be used to downsample the [SemEval 2020-Task 1](https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd/) corpora in English, German, and Swedish.  Requires all relevant data to have been installed using `download.sh`. 
* `download.sh` downloads the [SemEval 2020-Task 1](https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd/) and [Annotated Word Uses](https://www.ims.uni-stuttgart.de/data/wugs) datasets and places them in a directory format that `downsample.py` can understand.
* `evaluate.py` can be used to evaluate the performance of each model in `models/` on a single downsample.  If you wish to add the evaluation of a new model, it should be fairly straightforward to extend this script (see documentation in `evaluate.py` for how to do so).
* `experiments.py` is home to all experiments presented in our paper (paper link soon to come), and hopefully will provide a solid overview of how to either construct your own experiments with a new model or verify our results.
* `literary_analysis.py` is the script we used to conduct our case study in literary intertextual semantic change detection.  This script won't run without the digital copies of *The Wretched of the Earth* and *Scenes of Subjection*, but we provide it here for transparency and reproducibility.

## Install

The Python files in this directory have been tested using Python 3.10.8.  Use `pip install -r requirements.txt` to install all requirements.  If we are spinning up a rented GPU running a unix system with Python 3.10 already installed, we run the following commands to get the system setup.  Some of these may not be necessary depending on what resources are already installed on your machine.  Think of the following as a suggestion rather than an exact recipe to follow.

```bash
git clone https://github.com/jnehrenworth/small-corpora-scd.git
cd small-corpora-scd
sudo apt install unzip
chmod +x download.sh
./download.sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
sudo apt install build-essential
pip install -r requirements.txt
```

So that you do not feel cheated of explanation or inspired to run unknown commands, the above: clones this repository; downloads unzip (which you may already have installed); downloads the [SemEval 2020-Task 1](https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd/) and [Annotated Word Uses](https://www.ims.uni-stuttgart.de/data/wugs) datasets in English, German, and Swedish; installs `rustup` (necessary for the `tokenizers` package) and adds it to `PATH`; installs meta-packages with `build-essential` necessary for a number of different requirements (if you have `gcc` installed this may not be necessary); and then finally uses `pip` to install all necessary Python requirements.  

It may take some time to download the SemEval files, as even zipped they are quite large.

## Usage
So, you've gotten an environment spun up and you wish to use the system.  The following demonstrates a walkthrough and usage of each relevant file, but you may wish to jump directly to the `experiments.py` file if you wish to duplicate our results or duplicate our pipeline with a new model.  That file also shows how the pipeline works to integrate the different files in this repository. 

### Downsample
To create your own downsampled dataset, you will first have to download the [SemEval 2020-Task 1](https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd/) and [Annotated Word Uses](https://www.ims.uni-stuttgart.de/data/wugs) datasets using `sh download.sh` or `chmod +x download.sh; ./download.sh`.  This will download the SemEval and annotated uses datasets into the relevant directories.

Next, use the `downsample.py` script to create downsampled corpora of the desired token amount.  You can either overwrite the downsampled corpora I've provided, or give your own output directory.  In brief, you probably want to do something like `python downsample.py <language> -t <target_tokens> -w <path/to/out_dir>` for each `<language>` in "english", "german", and "swedish".  This will create a downsampled dataset in `<path/to/out_dir>` for the selected language where each corpus has about `<target>` number of words. See `python downsample.py -h` for more information about output directory structure and usage.

### Evaluate
Once you've created your own downsampled dataset, you can use `evaluate.py` to determine Spearman's $\rho$ for the models in `models/`.

The script by default reads from `data/downsampled`, but that can be overridden if you've placed your dataset in another directory.  You probably want to use it as `python evaluate.py <path/to/read_dir>`, where `<path/to/read_dir>` is the path you've input to the `downsample.py` program.  This script will train and evaluate each model's performance in `models/` on the corpora in `<path/to/read_dir>`.  See `python evaluate.py -h` for more information about this script.  **Note that model performance varies run to run.**  You will not get the exact same metrics we report in our paper with a single run of `evaluate.py`.

### Example Usage
If you want to, you can spin up a clean `conda` environment using something like `conda create -n small_corpora_scd python=3.10.8` and then `conda activate small_corpora_scd`.  The following code block assumes that you've already done that (or are using Python version 3.10), installed dependencies using `pip install -r requirements.txt`, and used `download.sh` to install relevant corpora.  What follows is an example of downsampling the SemEval datasets to 200k tokens and then evaluating each model in each language.

```console
user@small-corpora-scd% python downsample.py english -t 200000 -w data/downsample_example
(snipped)
user@small-corpora-scd% python downsample.py german -t 200000 -w data/downsample_example
(snipped)
user@small-corpora-scd% python downsample.py swedish -t 200000 -w data/downsample_example
(snipped)
user@small-corpora-scd% python evaluate.py data/downsample_example
(snipped)

========================= Evaluation Summary =========================

Reporting Spearman's rank correlation coefficient for each model in each language...

Model                English     German    Swedish
-----------------  ---------  ---------  ---------
UWB                 x.xxx     x.xxx       x.xxx
UG_Student_Intern   x.xxx     x.xxx       x.xxx
temporal_attention  x.xxx     x.xxx       
```

### Experiments
The file `experiments.py` was what we used to determine the results of sections 5.1, 5.2, and 5.3 in our paper.  As with every other file in this directory, before `experiments.py` is used the [SemEval 2020-Task 1](https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd/) and [Annotated Word Uses](https://www.ims.uni-stuttgart.de/data/wugs) corpora must be downloaded and all relevant requirements must be installed (see [Install](#install) for more information).  Run our experiments using `python experiments.py`.

There were three results presented in our paper:

  1. We downsampled the SemEval-2020 Task 1 corpora five hundred different times across all languages to 150k tokens and evaluated each model in the `models/` directory on these downsampled corpora (see our paper for more information on model selection).
  2. We measured the variability in Spearman’s $\rho$ for each model in `models/` across the 500 English downsamples.  
  3. We downsampled the SemEval-2020 Task 1 English corpora fifty different times to token targets from 250k to 6.25M (with jumps of 500k tokens) and evaluated each model in `models/` on these downsampled corpora.

The `experiments.py` file runs the experiments necessary to produce these results and saves the raw data to two csv files in `results/`: The first file, `stability_experiment.csv` houses data informing results 1. and 2., while the second file, `token_size_experiment.csv`, houses data informing result 3.  If additional models are added (see documentation in `evaluate.py` for how to do this), it should be quick to modify `experiments.py` to replicate the exact experiments we performed in our paper on the additional models.  See `experiments.py` for detailed documentation.  

Note that the csv files in `results/` use model names instead of author names as we do in our paper.  The following code can be used to map model names to author names when reading a result csv, assuming that one wishes to use pandas:
```Python
import pandas as pd

results_path = "results/stability_experiment.csv"

df = pd.read_csv(results_path)
df.replace("UG_Student_Intern", "Pömsl and Lyapin (2020)", inplace=True)
df.replace("UWB", "Pražák et al. (2020)", inplace=True)
df.replace("temporal_attention", "Rosin and Radinsky (2022)", inplace=True)
```

### Literary Analysis

As a case study for our proposed application—*literary intertextual semantic change detection*—we applied the Temporal Attention model presented by [Rosin and Radinsky (2022)](https://aclanthology.org/2022.findings-naacl.112/) to two books: *The Wretched of the Earth* by Franz Fanon and *Scenes of Subjection* by Saidiya Hartman.  The script `literary_analysis.py` is what we used to conduct this case study.  It lemmatizes and strips both books of punctuation, ranks non-stopwords that appeared more than 50 times in both books by degree of semantic change, and prints the resulting list.  It should be possible to make small changes to this file to create case studies involving other books.  See the if `__name__ = "__main__" ` guard in `literary_analysis.py` for slightly more information.

Without access to the digital copies of the two books we used, it may be that this file is not particularly helpful.  Still, we have released it in the hopes that it may aid in analyzing your own books and for full transparency.  Please reach out if you would like access to our private digitized copies of *The Wretched of the Earth* and *Scenes of Subjection* to fully reproduce our results.

## Important Caveats

- The `temporal_attention` model in `models/` will only run on a single GPU.  It is probably the case that enough debugging would enable it to run on multiple GPU instances, but if you are testing a new model and wish to run it across multiple GPUs it may be easiest to delete all evaluation of the `temporal_attention` model from `experiments.py`.

- As noted in [Literary Analysis](#literary-analysis), it will not be possible to run `literary_analysis.py` without access to digitized copies of Fanon's *The Wretched of the Earth* and Hartman's *Scenes of Subjection*.  Unfortunately, we cannot make these publicly available due to copyright.  Please reach out to obtain copies.

## Citation
If you found this project helpful to your research, please cite our paper using the following BibTeX: 

```
(forthcoming)
```