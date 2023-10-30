# small-corpora-scd
This repository accompanies our forthcoming paper, "Literary Intertextual Semantic Change Detection: Application and Motivation for Evaluating Models on Small Corpora" (permanent link available soon), appearing in the 4th International Workshop on Computational Approaches to Historical Language Change 2023 (LChange'23).  Here, you can find corpora, models, and testing scripts for the evaluation of semantic change detection (scd) models on small corpora.  See our paper for a general overview of the tasks and evaluation results, or read on for a repository primer.  If you have any questions, please don't hesitate to reach out (point of contact: Jackson Ehrenworth; jacksonehrenworth[@]gmail[.]com).  We welcome pull requests or other contributions.

If you found this project helpful to your research, please cite our paper using the following BibTeX: 

```
@inproceedings{ehrenworth2023literary,
  title={Literary Intertextual Semantic Change Detection: Application and Motivation for Evaluating Models on Small Corpora},
  author={Ehrenworth, Jackson and Keith, Katherine},
  booktitle={Proceedings of the 4th International Workshop on Computational Approaches to Historical Language Change},
  year={2023}
}
```

## Table of Contents

- [Repository Tour](#repository-tour) 
- [Install](#install)
- [Usage](#usage)
- [Models](#models)
- [Important Caveats](#important-caveats)

## Repository Tour

* `data/` houses downsampled, downloaded, and cached datasets.
* `models/` houses all models under evaluation.  Incidentally, if you wish to add your own model this would be the place to do it.  See the documentation attached to `evaluate.py` to get a sense for how one can do this.
* `results/` houses the raw data informing the results presented in our paper.
* `download.sh` downloads the [SemEval 2020-Task 1](https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd/) and [Annotated Word Uses](https://www.ims.uni-stuttgart.de/data/wugs) datasets and places them in a directory format that `downsample.py` can understand.
* `downsample.py` can be used to downsample the [SemEval 2020-Task 1](https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd/) corpora in English, German, and Swedish.  Requires all relevant data to have been installed using `download.sh`. 
* `evaluate.py` can be used to evaluate the performance of each model in `models/` on a single downsample.  If you wish to add the evaluation of a new model, it should be fairly straightforward to extend this script (see documentation in `evaluate.py` for how to do so).
* `experiments.py` is home to all experiments presented in our paper (paper link soon to come), and will provide a solid overview of how to either construct your own experiments with a new model or verify our results.
* `literary_analysis.py` is the script we used to conduct our case study in literary intertextual semantic change detection.  This script won't run without the digital copies of *The Wretched of the Earth* and *Scenes of Subjection*—which we cannot make publicly available due to copyright—but we provide it here for transparency and reproducibility.  Please contact us if interested in obtaining our digital copies for purposes of research reproducibility.

## Install

The Python files in this directory have been tested using Python 3.10.8.  Use `pip install -r requirements.txt` to install all requirements.  If we are spinning up a rented GPU running a unix system with Python 3.10 already installed, we run the following commands to get the system setup. 

```bash
git clone https://github.com/jnehrenworth/small-corpora-scd.git
cd small-corpora-scd
sudo apt install unzip
chmod +x download.sh
./download.sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain=1.72.1 -y
source "$HOME/.cargo/env"
export RUSTUP_TOOLCHAIN=1.72.1
sudo apt install build-essential
pip install -r requirements.txt
```

So that you do not feel cheated of explanation or inspired to run unknown commands, the above: clones this repository; downloads unzip (which you may already have installed); downloads the [SemEval 2020-Task 1](https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd/) and [Annotated Word Uses](https://www.ims.uni-stuttgart.de/data/wugs) datasets in English, German, and Swedish; installs an older version of `rustup` (necessary for the `tokenizers` package) and adds it to `PATH`; installs meta-packages with `build-essential` necessary for a number of different requirements (if you have `gcc` installed this may not be necessary); and then finally uses `pip` to install all necessary Python requirements.  

It is necessary to use a slightly older version of Rust—1.72.1—when running `pip install -r` as Rust >=1.73.0 will fail to build the tokenizers version necessary for some of the models in `models/`.  See, for instance, the StackOverflow post: [Cargo Rustc Failed with Code 101 Could Not Build Wheel for Tokenizers](https://stackoverflow.com/questions/77265938/cargo-rustc-failed-with-code-101-could-not-build-wheels-for-tokenizers-which).  This is done in the above script by setting the RUSTUP_TOOLCHAIN environment variable and installing Rust version 1.72.1.

It may take some time to download the SemEval files, as even zipped they are quite large.

## Usage
This section demonstrates the usage of each relevant file provided in this repository.  The prerequisite for each command run under this heading is that the environment, packages, and necessary datasets have all been setup or downloaded.  You may wish to jump directly to the `experiments.py` file to understand how to duplicate our results or duplicate our pipeline with a new model.  That file also shows how the pipeline works to integrate the different files in this repository. 

### Example Usage
If you want to, you can spin up a clean `conda` environment using something like `conda create -n small_corpora_scd python=3.10.8` and then `conda activate small_corpora_scd`.  The following code block assumes that you've already done that (or are using Python version 3.10), installed dependencies using `pip install -r requirements.txt`, and used `download.sh` to install relevant corpora.  What follows is an example of downsampling the SemEval datasets to 200k tokens and then evaluating each model in each language.  We've snipped the logging output of each of these commands for brevity’s sake.

```console
user@small-corpora-scd% python downsample.py english -t 200000 -w data/downsample_example
(output snipped)
user@small-corpora-scd% python downsample.py german -t 200000 -w data/downsample_example
(output snipped)
user@small-corpora-scd% python downsample.py swedish -t 200000 -w data/downsample_example
(output snipped)
user@small-corpora-scd% python evaluate.py data/downsample_example
(partial output snipped)

========================= Evaluation Summary =========================

Reporting Spearman's rank correlation coefficient for each model in each language...

Model                English     German    Swedish
-----------------  ---------  ---------  ---------
UWB                 x.xxx     x.xxx       x.xxx
UG_Student_Intern   x.xxx     x.xxx       x.xxx
temporal_attention  x.xxx     x.xxx       
```

### Downsample
Use the `downsample.py` script to create downsampled corpora of the desired token amount.  You can either overwrite the downsampled corpora we've provided, or give your own output directory.  Example usage:

```console
user@small-corpora-scd% python downsample.py <language> -t <target_tokens> -w <path/to/out_dir>
```

In brief, you probably want to do run this command for each `<language>` in "english", "german", and "swedish".  This will create a downsampled dataset in `<path/to/out_dir>` for the selected language where each corpus has about `<target>` number of words. 

See `python downsample.py -h` for more information about output directory structure and usage.

### Evaluate
Once you've created your own downsampled dataset, you can use `evaluate.py` to determine Spearman's $\rho$ for the models in `models/`.

The script by default reads from `data/downsampled`, but that can be overridden if you've placed your dataset in another directory.  Example usage:

```console
user@small-corpora-scd% python evaluate.py <path/to/read_dir>
```

Where `<path/to/read_dir>` is the path you've input to the `downsample.py` program.  This script will train and evaluate each model's performance in `models/` on the corpora in `<path/to/read_dir>`.  See `python evaluate.py -h` for more information about this script.  **Note, due to the stochastic nature of  in the models, performance will vary run to run.**  You will not get the exact same metrics we report in our paper with a single run of `evaluate.py`.

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

## Models

We evaluated three models that present a range of different architectures, from static (non-contextual) embeddings to contextual embeddings, that are, to our knowledge, among the currently the highest performing open-source models for unsupervised semantic change detection.  In the following list, each model name contains a link to the model's GitHub repository.  The citation is a link to the paper where the model was presented.

1. The [UWB](https://github.com/pauli31/SemEval2020-task1) winning submission on Subtask 1 in the SemEval 2020 Task 1 competition proposed by [Pražák et al. (2020)](https://aclanthology.org/2020.semeval-1.30/).

   The authors train word2vec style embeddings using Skip-Gram with Negative Sampling (SGNS), align them using orthogonal Procrustes, and then use cosine distance to compare aligned embeddings.  
2. The [UG_Student_Intern](https://github.com/mpoemsl/circe) winning submission on Subtask 2 in the SemEval 2020 Task 1 competition proposed by [Pömsl and Lyapin (2020)](https://aclanthology.org/2020.semeval-1.21/).  

    The authors train word2vec style embeddings using SGNS, align them using orthogonal Procrustes, and then take Euclidean distance as their metric when comparing aligned embeddings.  Note that although they describe ensemble and context-based models in their paper, their winning submission used word2vec context-free embeddings and is what we've chosen to evaluate.
3. The [temporal_attention](https://github.com/guyrosin/temporal_attention) model presented by [Rosin and Radinsky (2022)](https://arxiv.org/abs/2202.02093).

    This is the highest performing open-source contextualized semantic shift detection model on Subtask 2 we are aware of ([Montanelli and Periti, 2023](https://arxiv.org/abs/2304.01666)).  The authors propose a temporal self-attention mechanism as a modification to the standard transformers architecture. They use a pre-trained BERT model, fine-tune it on diachronic corpora using their proposed temporal attention mechanism, and then create timespecific representations of target words by extracting and averaging hidden-layer weights. These representations are then averaged at the token level and compared using cosine similarity.

We pulled the latest version of each model's GitHub on May 2nd, 2023.

## Important Caveats

- The `temporal_attention` model in `models/` will only run on a single GPU.  It is probably the case that enough debugging would enable it to run on multiple GPU instances, but if you are testing a new model and wish to run it across multiple GPUs it may be easiest to delete all evaluation of the `temporal_attention` model from `experiments.py`.

- As noted in [Literary Analysis](#literary-analysis), it will not be possible to run `literary_analysis.py` without access to digitized copies of Fanon's *The Wretched of the Earth* and Hartman's *Scenes of Subjection*.  Unfortunately, we cannot make these publicly available due to copyright.  Please reach out to obtain copies.