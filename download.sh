#!/bin/sh

download_semeval () {
    download_link=$1
    short_name=$2
    long_name=$3
    corpus1_name=$4
    corpus2_name=$5

    echo "\n========== Downloading $long_name SemEval 2020-Task 1 Dataset ==========\n"

   # load zip files
    curl $download_link -o temp/$long_name-semeval.zip

    echo "\n=== Unzipping ===\n"

    # unzip everything
    unzip temp/$long_name-semeval.zip -d temp
    gunzip temp/semeval2020_ulscd_$short_name/corpus1/lemma/$corpus1_name.txt.gz
    gunzip temp/semeval2020_ulscd_$short_name/corpus2/lemma/$corpus2_name.txt.gz

    # make semeval directory
    mkdir data/semeval/$long_name

    # move relevant files
    mv temp/semeval2020_ulscd_$short_name/corpus1/lemma/$corpus1_name.txt data/semeval/$long_name/$corpus1_name.txt
    mv temp/semeval2020_ulscd_$short_name/corpus2/lemma/$corpus2_name.txt data/semeval/$long_name/$corpus2_name.txt
    mv temp/semeval2020_ulscd_$short_name/targets.txt data/semeval/$long_name/targets.txt
    mv temp/semeval2020_ulscd_$short_name/truth data/semeval/$long_name

    echo "\n========== Download Successful ==========\n"
}

download_dwug () {
    download_link=$1
    short_name=$2
    long_name=$3

    echo "\n========== Downloading $long_name SemEval 2020-Task 1 Annotated Uses ==========\n"

    curl $download_link -o temp/$long_name-uses.zip

    echo "\n=== Unzipping ===\n"

    unzip temp/$long_name-uses.zip -d temp
    
    mkdir data/annotated_uses/$long_name

    cd temp/dwug_$short_name/data
    for FILE in *; do 
        lower="$(echo $FILE | tr '[:upper:]' '[:lower:]')"

        mv $FILE/uses.csv ../../../data/annotated_uses/$long_name/$lower.csv
    done

    cd ../../..

    echo "\n========== Download Successful ==========\n"
}

# make temporary directory to store zip garbage
mkdir temp

# semeval download
# latin is not currently supported so there's no reason to download it
mkdir data/semeval

download_semeval https://www2.ims.uni-stuttgart.de/data/sem-eval-ulscd/semeval2020_ulscd_eng.zip eng english ccoha1 ccoha2
download_semeval https://www2.ims.uni-stuttgart.de/data/sem-eval-ulscd/semeval2020_ulscd_ger.zip ger german dta bznd
download_semeval https://zenodo.org/records/3730550/files/semeval2020_ulscd_swe.zip?download=1 swe swedish kubhist2a kubhist2b
# download_semeval https://zenodo.org/record/3734089/files/semeval2020_ulscd_lat.zip?download=1 lat latin LatinISE1 LatinISE2

# annotated uses download
mkdir data/annotated_uses

download_dwug https://zenodo.org/records/7387261/files/dwug_en.zip?download=1 en english
download_dwug https://zenodo.org/records/7441645/files/dwug_de.zip?download=1 de german
download_dwug https://zenodo.org/records/7389506/files/dwug_sv.zip?download=1 sv swedish
# download_dwug https://zenodo.org/record/5255228/files/dwug_la.zip?download=1 la latin

# remove temp directory
rm -rf temp