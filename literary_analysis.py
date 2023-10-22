import nltk
import shutil
from collections import Counter
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from pprint import pprint
from string import punctuation

from evaluate import preprocess_for_TA, populate, path_rules_TA, train_ta, _get_semantic_changes


#########################################
# Helpers                               #
#########################################

def to_pos(nltk_tag: str):
    """Maps nltk Treebank tags to WordNet part of speech.

    See https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html,
    https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python,
    or https://www.holisticseo.digital/python-seo/nltk/lemmatize for more details on
    lemmatization with nltk.  This function is almost directly ripped from the SO post.
    """
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    return None

def lemmatize_sentence(sentence: str, lemmatizer: WordNetLemmatizer, punc: set[str]) -> str:
    """Lemmatizes `sentence` with `lemmatizer`, accounting for part of speech and ignoring any
    punctuation in `punc`.  
    
    Lemmatization that accounts for part of speech isn't provided entirely out of the box with nltk.  One must part of speech tag each word and then pass that information into the WordNetLemmatizer.
    See https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python 
    for more information.
    """
    treebank_tags = nltk.pos_tag([word for word in word_tokenize(sentence.lower().strip()) if word not in punc])
    pos_tags = [(word, to_pos(nltk_tag)) for word, nltk_tag in treebank_tags]
    return " ".join([
        lemmatizer.lemmatize(word, tag) if tag is not None else word for word, tag in pos_tags
    ])

def lemmatize_paragraph(paragraph: str, lemmatizer: WordNetLemmatizer, punc: set[str]) -> str:
    """Returns a newline separated list of sentences that have been lemmatized.

    `paragraph` is sentence-tokenized using `nltk.sent_tokenize` and then each sentence
    is word-tokenized using `nltk.word_tokenize` and lemmatized with `lemmatizer` accounting 
    for part of speech.  Any punctuation included in the `punc` set will be stripped from
    the returned string.

    We return a newline separated string because the temporal_attention model expects each sentence
    to be on its own line (as in the SemEval corpora). 
    """
    return "\n".join(
        lemmatize_sentence(sentence, lemmatizer, punc) for sentence in sent_tokenize(paragraph)
    )

def clean(book_path: str):
    """Lemmatizes and strips punctuation from the .txt file pointed to by `book_path`, rewriting
    each sentence on a newline.
    """
    lemmatizer = WordNetLemmatizer()
    # There are a couple other strange artifacts that we have to add for removal
    # in addition to string.punctuation.
    punc = set(punctuation + "“”’s")
    punc.add("``")
    punc.add("'s")

    with open(book_path, "r+") as f:
        lines = [lemmatize_paragraph(line, lemmatizer, punc) for line in f.readlines() if not line.isspace()]
        f.seek(0)
        f.write("\n".join(lines))
        f.truncate()

def get_target_types(book_path: str, threshold: int = 50) -> dict[str, int]:
    """Returns a mapping of non-stopwords to the number of times they appeared in the
    file pointed to by `book_path`.

    Any type that has fewer than `threshold` instances appearing is removed from the
    returned counter.  This function does no cleaning, so any preprocessing should
    be done before this function is called.  Any stopwords (defined by `nltk.corpus.stopwords`)
    are ignored.
    """
    types = Counter()

    with open(book_path) as book:
        for line in book:
            for word in line.split():
                types[word] += 1

    types_above_threshold = {type: count for type, count in types.items() if count > threshold}
    relevant_types = {k: v for k, v in types_above_threshold.items() if k not in stopwords.words("english")}
    return relevant_types

def write_targets(book_dir: str, book1_path: str, book2_path: str):
    """Writes a list of target types to {book_dir}/english/targets.txt sorted in 
    descending order by the number of times they appear.

    Target types are non-stopwords that appeared more than 50 times in both
    books.  We write to the seemingly strange {book_dir}/english/targets.txt location 
    because that is the same format as the SemEval corpora targets use.
    """
    book1_types = get_target_types(book1_path)
    book2_types = get_target_types(book2_path)
    target_types = {k: book1_types[k] + book2_types[k] for k in book1_types.keys() & book2_types.keys()}
    sorted_types = {k: v for k, v in sorted(target_types.items(), key=lambda item: item[1], reverse=True)}

    print("Targets, sorted by popularity: ")
    pprint(sorted_types, sort_dicts=False)
    with open(f"{book_dir}/english/targets.txt", "w") as targets:
        targets.write("\n".join(type for type in sorted_types))


#########################################
# Drivers                               #
#########################################

def run_analysis(book_dir: str, book1_path: str, book2_path: str):
    """Lemmatizes and strips both books of punctuation, then ranks non-stopwords that appeared 
    more than 50 times in both books via the temporal_attention model by degree of semantic change
    and prints this list to stdout.
    """
    clean(book1_path)
    clean(book2_path)
    write_targets(book_dir, book1_path, book2_path)
    
    # The `print()` statements are just to make things a little prettier
    print() 
    shutil.rmtree("models/temporal_attention/data/semeval_eng") 
    populate(path_rules_TA, book_dir)
    preprocess_for_TA()
    print()

    train_ta("english")
    change_scores = _get_semantic_changes("english")
    ranked_words = {k: v for k, v in sorted(change_scores.items(), key=lambda item: item[1], reverse=True)}
    print()

    pprint(ranked_words, sort_dicts=False)


if __name__ == "__main__":
    # Note that it will not be possible to directly run this file without the following
    # directory structure.  We are not currently aware of any publicly accessible .txt copies 
    # of Scenes of Subjection or The Wretched of the Earth.  Please reach out if you want access
    # to our copies.
    # 
    # Theoretically, running this file on different books should be as simple as placing 
    # different books into the data/literary_analysis folder and then changing the .txt
    # file names defining the path to book1/2.  But we haven't tested this, so no promises
    book_dir = "data/literary_analysis"
    book1_path = f"{book_dir}/english/Scenes of Subjection_2022.txt"
    book2_path = f"{book_dir}/english/The Wretched of the Earth_2021.txt"
    run_analysis(book_dir, book1_path, book2_path)