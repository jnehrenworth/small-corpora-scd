# small-corpora-ssd
This repository houses corpora, models, and testing scripts for the evaluation of unsupervised semantic shift detection (ssd) models on small corpora. 

## Overview
The standard dataset against which semantic shift detection models are evaluated was presented in [SemEval-2020 Task 1: Unsupervised Lexical Semantic Change Detection](http://dx.doi.org/10.18653/v1/2020.semeval-1.1) by Schlechtweg et al. The authors released corpora in four different languages—English, German, Latin, and Swedish—each of which are bifurcated diachronically at some time period. For each pair of corpora in a given language, call them $C_1$ and $C_2$, Schlechtweg et al. present two subtasks:

1. Binary classification: the goal is to predict for a target word whether semantic shift has occurred between $C_1$ and $C_2$.
2. Degree of semantic shift: the goal is to determine the amount of semantic shift that a list of words have undergone between $C_1$ and $C_2$ by proxy of ranking them according to their degree of semantic shift (e.g., "gay" has changed more than "cell" which has changed more than "peer").

This project is exclusively focused on Subtask 2.  For one, it seems intuitively likely (and the [SemEval-2020 Task 1](http://dx.doi.org/10.18653/v1/2020.semeval-1.1) results tend to bear this out) that good performance on Subtask 2 is indicative of good performance on Subtask 1.  It also seems to me as if Subtask 2 captures more about the subtle movement of language that is most applicable to those interested in using semantic shift detection models on small corpora.  For instance, a polysemous word may not experience a binary sense change between $C_1$ and $C_2$ but may yet shift from primarily one sense type to another.  As this kind of subtle semiotic shift is of most interest to those in the digital humanities, it is what I have focused on.  I'm interested in extending this project to Subtask 1 once I have more time.  

Subtask 2 is evaluated using Spearman’s rank-order correlation coefficient $\rho$ against a ground truth ranked list (for more information about how ground truth rankings were created see Schlechtweg et al.).  Scores are bounded between $[-1, 1]$, with a score of $1$ indicating perfect correlation between predicted and true ranking and a score of $-1$ in antipode indicating a predicted ranking that is the complete opposite of the true ranking.

For those interested in applying ssd models to small corpora (e.g., single authored corpora), there is a need for a SemEval style evaluation of models on smaller corpora.  Rather than introducing novel corpora, this project is focused on downsampling the [SemEval-2020 Task 1 corpora](https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd/) while still preserving the ground truth data the authors used an expensive manual annotation process to obtain.  Randomly sampling a certain number of sentences from each corpus until a target token limit is met would destroy the ground truth data.  In order to preserve it, then, one can cross-reference each corpus with the raw annotated word uses[^1] presented by the same authors in [DWUG: A large Resource of Diachronic Word Usage Graphs in Four Languages](https://aclanthology.org/2021.emnlp-main.567.pdf).  This repository provides scripts that handle the downsampling process and evaluate some of the state-of-the-art models on the smaller downsampled corpora.  See [Usage](#usage) for more information on how to use the scripts in this repository, or [Repository Tour](#repository-tour) for, as perhaps one would expect, a tour of the repository.

Currently, the models that evaluation is supported for are:

1. The [UWB](https://github.com/pauli31/SemEval2020-task1) winning submission on Subtask 1 in the SemEval 2020 Task 1 competition.  

   The authors train word2vec style embeddings using Skip-Gram with Negative Sampling (SGNS), align them using Canonical Correlation Analysis and orthogonal Procrustes, and then use cosine distance to compare aligned embeddings. See [UWB at SemEval-2020 Task 1](https://aclanthology.org/2020.semeval-1.30/) by Pražák et al. for more information on their model's architecture.
2. The [UG_Student_Intern](https://github.com/mpoemsl/circe) winning submission on Subtask 2 in the SemEval 2020 Task 1 competition.    

    The authors train word2vec style embeddings using SGNS, align them using orthogonal Procrustes, and then take Euclidean distance as their metric when comparing aligned embeddings.  See [CIRCE at SemEval-2020 Task 1: Ensembling Context-Free and Context-Dependent Word Representations](https://aclanthology.org/2020.semeval-1.21/) by Pömsl & Lyapin for more information on their model's architecture.  Note that although they describe ensemble and context-based models in their paper, their winning submission used word2vec context-free embeddings and is what I've chosen to evaluate.
3. The [temporal_attention](https://github.com/guyrosin/temporal_attention) model presented in [A model Temporal Attention for Language Models](https://arxiv.org/abs/2202.02093).  

    The authors propose a temporal self-attention mechanism as a modification to the standard transformers architecture.  They use a pretrained BERT model (bert-base-uncased: 12 layers, 768 hidden dimensions, 110M parameters), train it on the temporal corpora using their proposed temporal-attention mechanism, and then create time-specific representations of target words by extracting and averaging hidden-layer weights.  These representations are then compared using cosine similarity as a measure of semantic change.  See [A model Temporal Attention for Language Models](https://arxiv.org/abs/2202.02093) for more information.  To my knowledge, this is the current state-of-the-art model when evaluated on Subtask 2.

I selected these models because they present a range of different architectures, from context-free dense embeddings to context-based language model type embeddings, and are, at least to my knowledge, representative of the highest performing current models we have for unsupervised semantic shift detection.  I've used the models essentially as-is from their respective GitHubs (I've had to make some modifications here and there, but only to fix some small issues—none of my changes altered any of the models architecture or salient features, they were essentially all bug or version fixes).

If you're only interested in the results, here is what I've found when evaluating these models against corpora that have been downsampled to 150k tokens (I haven't yet gotten around to evaluating the `temporal_attention` model, but look for that shortly).[^2]

### Results

#### Downsampled Datasets:

| Model              | English  | German     | Swedish    |
|--------------------|:--------:|:----------:|:----------:|
| UWB                | 0.061    | 0.396      | 0.299      |
| UG_Student_Intern  | 0.129    | -0.043     | -0.053     |
| temporal_attention | x.xxx    | x.xxx      | x.xxx      |

#### Original Results on SemEval 2020-Task 1:

| Model              | English  | German     | Swedish    |
|--------------------|:--------:|:----------:|:----------:|
| UWB                | 0.367    | 0.697      | 0.604      |
| UG_Student_Intern  | 0.422    | 0.725      | 0.547      |
| temporal_attention | 0.520    | 0.763      | x.xxx[^3]  |

## Usage

The Python files in this directory have been tested using Python 3.10.8.  Use `pip install -r requirements.txt` to install all requirements.  

### Downsample
To create your own downsampled dataset, you will first have to download the [SemEval 2020-Task 1](https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd/) and [Annotated Word Uses](https://www.ims.uni-stuttgart.de/data/wugs) datasets.  If you are on a system that supports `curl`, run `chmod +x download_semeval.sh` and then `./download_semeval.sh`.  This will download the SemEval and Uses datasets into the relevant directories.  Note that these are quite large files, even when zipped, and may take some time to download.  

Next, use the `downsample.py` script to create downsampled corpora of the desired token amount.  You can either overwrite the downsampled corpora I've provided, or give your own output directory.  In brief, you probably want to do something like `python downsample.py <language> -t <target_tokens> -w <path/to/out_dir>` for each `<language>` in "english", "german", and "swedish".  This will create a downsampled dataset in `<path/to/out_dir>` for the selected language where each corpus has about `<target>` number of words. See `python downsample.py -h` for more information about output directory structure and usage.

### Evaluate
So, you've either created your own downsampled dataset or you want to validate my results.  You can use `evaluate.py` for this purpose.  The script by default reads from `data/downsampled`, but that can be overridden if you've placed your downsampled dataset in another directory.  You probably want to use it as `python evaluate.py <path/to/read_dir>`, where `<path/to/read_dir>` is either path you've input to the `downsample.py` program or can be left off to read from `data/downsampled`.  This script will train and evaluate each model's performance on Subtask 2 on the downsampled corpora.  See `python evaluate.py -h` for more information about this script.

### Example Usage

If you want to, you can spin up a clean `conda` environment using something like `conda create -n small_corpora_ssd python=3.10.8` and then `conda activate small_corpora_ssd`.  The following code block assumes that you've already done that (or are using Python version 3.10) and then installed dependencies using `pip install -r requirements.txt`.  

```console
user@small-corpora-ssd% chmod +x download.sh
user@small-corpora-ssd% ./download.sh

========== Downloading english SemEval 2020-Task 1 Dataset ==========

  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 57.9M  100 57.9M    0     0   313k      0  0:03:09  0:03:09 --:--:--  364k

=== Unzipping ===

Archive:  temp/english-semeval.zip
   creating: temp/semeval2020_ulscd_eng/
   creating: temp/semeval2020_ulscd_eng/corpus2/
   creating: temp/semeval2020_ulscd_eng/corpus2/lemma/
  inflating: temp/semeval2020_ulscd_eng/corpus2/lemma/ccoha2.txt.gz  
   creating: temp/semeval2020_ulscd_eng/corpus2/token/
  inflating: temp/semeval2020_ulscd_eng/corpus2/token/ccoha2.txt.gz  
   creating: temp/semeval2020_ulscd_eng/truth/
  inflating: temp/semeval2020_ulscd_eng/truth/graded.txt  
  inflating: temp/semeval2020_ulscd_eng/truth/binary.txt  
  inflating: temp/semeval2020_ulscd_eng/README.md  
   creating: temp/semeval2020_ulscd_eng/corpus1/
   creating: temp/semeval2020_ulscd_eng/corpus1/lemma/
  inflating: temp/semeval2020_ulscd_eng/corpus1/lemma/ccoha1.txt.gz  
   creating: temp/semeval2020_ulscd_eng/corpus1/token/
  inflating: temp/semeval2020_ulscd_eng/corpus1/token/ccoha1.txt.gz  
  inflating: temp/semeval2020_ulscd_eng/README.html  
  inflating: temp/semeval2020_ulscd_eng/targets.txt  

========== Download Successful ==========


========== Downloading german SemEval 2020-Task 1 Dataset ==========
(sniped)

user@small-corpora-ssd% python downsample.py english -t 200000 -w data/downsample_example
========================= Getting Target Words =========================

Gathering targets from data/semeval/english/targets.txt

 TARGETS:
--------------------
  attack_nn, bag_nn, ball_nn, bit_nn, chairman_nn, circle_vb, contemplation_nn,
  donkey_nn, edge_nn, face_nn, fiction_nn, gas_nn, graft_nn, head_nn, land_nn, lane_nn,
  lass_nn, multitude_nn, ounce_nn, part_nn, pin_vb, plane_nn, player_nn, prop_nn,
  quilt_nn, rag_nn, record_nn, relationship_nn, risk_nn, savage_nn, stab_nn, stroke_vb,
  thump_nn, tip_vb, tree_nn, twist_nn, word_nn

========================= Getting Target Uses =========================

Searching for annotated uses from data/semeval/english/ccoha1.txt...

100 %|██████████████████████████████████████████| 253644/253644 [00:01<00:00, 144109.86it/s]

Searching for annotated uses from data/semeval/english/ccoha2.txt...

100 %|██████████████████████████████████████████| 353692/353692 [00:01<00:00, 185511.51it/s]

========================= Downsampling =========================

Downsampling from data/semeval/english/ccoha1.txt into data/downsample_example/english/ccoha1.txt

100 %|██████████████████████████████████████████| 253644/253644 [00:00<00:00, 1125819.52it/s]

   Downsample Summary

Category          Tokens
--------------  --------
Annotated Uses    138455
Random Samples     61559
Total             200014

Downsampling from data/semeval/english/ccoha2.txt into data/downsample_example/english/ccoha2.txt

100 %|██████████████████████████████████████████| 353692/353692 [00:00<00:00, 1376919.57it/s]

   Downsample Summary

Category          Tokens
--------------  --------
Annotated Uses     94580
Random Samples    105425
Total             200005

========================= Verifying Target Uses in Downsampled =========================

Verifying annotated uses in data/downsample_example/english/ccoha1.txt...

100 %|██████████████████████████████████████████| 6075/6075 [00:00<00:00, 105720.33it/s]

Verifying annotated uses in data/downsample_example/english/ccoha2.txt...

100 %|██████████████████████████████████████████| 9197/9197 [00:00<00:00, 140722.06it/s]

Success: Downsample completed and verified in English, see tables for summary statistics.
user@small-corpora-ssd% python downsample.py german -t 200000 -w data/downsample_example
(snipped)
user@small-corpora-ssd% python downsample.py swedish -t 200000 -w data/downsample_example
(snipped)
user@small-corpora-ssd% python evaluate.py data/downsample_example

========================= Populating Dataset Paths =========================

Populating for UG_Student_Intern
--------------------------------

Writing data/downsample_example/german/bznd.txt to models/UG_Student_Intern/datasets/de-semeval/c2.txt
Writing data/downsample_example/german/dta.txt to models/UG_Student_Intern/datasets/de-semeval/c1.txt
Writing data/downsample_example/english/ccoha1.txt to models/UG_Student_Intern/datasets/en-semeval/c1.txt
Writing data/downsample_example/english/ccoha2.txt to models/UG_Student_Intern/datasets/en-semeval/c2.txt
Writing data/downsample_example/swedish/kubhist2b.txt to models/UG_Student_Intern/datasets/sw-semeval/c2.txt
Writing data/downsample_example/swedish/kubhist2a.txt to models/UG_Student_Intern/datasets/sw-semeval/c1.txt

Success: UG_Student_Intern datasets populated

(snipped)

========================= Evaluating Models =========================

Evaluating UG_Student_Intern
----------------------------

Predicting with context-free model for dataset en-semeval ...
Experiment data will be stored in models/UG_Student_Intern/experiments/context-free_en-semeval/ ...
Preprocessing texts ...
Training Word2Vec ...
Running command: word2vec -train models/UG_Student_Intern/experiments/context-free_en-semeval/preprocessed_texts/c1.txt -output models/UG_Student_Intern/experiments/context-free_en-semeval/word_representations/c1.vec -size 300 -window 10 -sample 1e-3 -hs 0 -negative 1 -threads 12 -iter 5 -min-count 0 -alpha 0.025 -debug 2 -binary 0 -cbow 0
Starting training using file models/UG_Student_Intern/experiments/context-free_en-semeval/preprocessed_texts/c1.txt
Vocab size: 14948
Words in train file: 206087
Alpha: 0.002305  Progress: 91.75%  Words/thread/sec: 25.95k  
Running command: word2vec -train models/UG_Student_Intern/experiments/context-free_en-semeval/preprocessed_texts/c2.txt -output models/UG_Student_Intern/experiments/context-free_en-semeval/word_representations/c2.vec -size 300 -window 10 -sample 1e-3 -hs 0 -negative 1 -threads 12 -iter 5 -min-count 0 -alpha 0.025 -debug 2 -binary 0 -cbow 0
Starting training using file models/UG_Student_Intern/experiments/context-free_en-semeval/preprocessed_texts/c2.txt
Vocab size: 20146
Words in train file: 209200
Alpha: 0.002365  Progress: 91.50%  Words/thread/sec: 27.40k  
Aligning embeddings ...
Comparing context-free representations ...
Finished experiment. Prediction can be found in models/UG_Student_Intern/experiments/context-free_en-semeval/prediction.tsv.

(snipped)

========================= Evaluation Summary =========================

Reporting Spearman's rank correlation coefficient for each model in each language...

Model                English     German    Swedish
-----------------  ---------  ---------  ---------
UWB                 0.05      0.343       0.2
UG_Student_Intern   0.139664  0.0736592   0.181709
```

## Repository Tour

* `evaluate.py` is a script that can be used to evaluate the performance of each model described above.  If you wish to add the evaluation of a new model, it should be fairly straightforward to extend this script (see documentation in `evaluate.py` for how to do so).
* `downsample.py` is a script that can be used to create your own downsampled datasets, assuming that you've installed all relevant data using `download.sh`.
* `download.sh` is a shell script that downloads the [SemEval 2020-Task 1](https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd/) and [Annotated Word Uses](https://www.ims.uni-stuttgart.de/data/wugs) datasets and places them in a directory format that `downsample.py` can understand.
* `data/` houses all downsampled and downloaded datasets/corpora.
* `models/` houses the models that are under evaluation.

[^1]: When I say "word uses" I mean the sentences that use some target word (e.g., attack) that were manually annotated.  For example, the sentence, "As the stranger fell to the earth under an attack so impetuous and unexpected, he uttered an exclamation in which Juan recognized the language of Mexico" was one of the manually annotated sentences for the target word "attack" so it must be included.  See [DWUG: A large Resource of Diachronic Word Usage Graphs in Four Languages](https://aclanthology.org/2021.emnlp-main.567.pdf) for more information.

[^2]: Note that there is some variability run to run based on the stochastic nature of these models, so don't expect to get the *exact* same results as I have put in this table when you do your evaluation.  These values are based on the average of ten different runs calling the `evaluation.py` script.

[^3]: Note that Rosin and Radinsky did not evaluate their model on the Swedish corpora, perhaps due to the computation time that would be required (the Swedish corpora are by far the largest of the SemEval 2020 corpora).