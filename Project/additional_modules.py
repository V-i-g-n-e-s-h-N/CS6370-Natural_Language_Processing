import math

from nltk.corpus import wordnet
from gensim import corpora, models, similarities #gensim==4.4.0, scipy==1.17.1, sentence_transformers
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from inflectionReduction import InflectionReduction
import re
from typing import List, Tuple
import numpy as np #numpy 2.4.4
from pathlib import Path
import json
from scipy.stats import wilcoxon
import pandas as pd
from io import StringIO
import pickle
from collections import defaultdict

enclosing_dirpath = Path(__file__).resolve().parent
model_path = enclosing_dirpath / "models" / "Model_for_ESA"
thesaurus_path = enclosing_dirpath / "models" / "NASA_thesaurus_CSV.txt"
out_dir_path = enclosing_dirpath / "output"

class GensimLSI():
    def __init__(self, docs: List[List[str]], num_topics: int, doc_ids:List[int]):
        """
        Creates a Latent Semantic Indexer for the given corpus

        Parameters:
        ------------
        docs: List[List[str]]
            A list of documents (lists) with each document being a list of tokens(strings) .
        num_topics: int
            The number of latent topics (internally chooses the best k eigenvalues)
        """
        self.dicti =  corpora.Dictionary(docs)
        self.corpus = [self.dicti.doc2bow(doc) for doc in docs]
        self.doc_ids = doc_ids
        self.tfidf = models.TfidfModel(self.corpus, smartirs='ltc')
        self.lsi = models.LsiModel(self.tfidf[self.corpus], id2word=self.dicti, num_topics=num_topics)
        self.corpus_tfidf = self.tfidf[self.corpus]
        self.index = similarities.MatrixSimilarity(self.lsi[self.corpus_tfidf])

    def inspect_topics(self):
        """
        Returns the discovered latent topics
        """
        topics = []
        for topic in self.lsi.print_topics():
            topics.append(topic)
            
        return topics

    def return_similarity_matrix(self):
        """
        Returns similarity matrix
        """
        return self.index
    
    def retrieve_docs(self, query: List[str]) -> List[Tuple[int, float]]:
        """
        Outputs similarity values for a query w.r.t the supplied corpus
        """
        query_bow = self.dicti.doc2bow(query)
        query_tfidf = self.tfidf[query_bow]   # need self.tfidf
        query_lsi = self.lsi[query_tfidf]
        sims = self.index[query_lsi]
        sims_sorted = sorted(enumerate(sims), key=lambda x: (-x[1], x[0]))
        return [(self.doc_ids[pos_id], score) for pos_id, score in sims_sorted]

class ESAWordRepn():
    def __init__(self, docs:List[List[str]], doc_ids:List[int]):
        """
        An ESA implementation based on GloVe (similar to word2vec)
        Must use lemmatizer
        """
        self.model = SentenceTransformer(str(model_path)) 
        self.encoded_docs = [self.model.encode(" ".join(doc)) for doc in docs] # List[np.ndarray]
        self.doc_ids = doc_ids

    
    def retrieve_docs(self, query: List[str]) -> List[Tuple[int, float]]:
        """
        Outputs similarity values for a query w.r.t the supplied corpus
        """
        encoded_query = self.model.encode(" ".join(query))
        encoded_docs = np.stack(self.encoded_docs, axis=0)
        sims = (np.dot(encoded_docs, encoded_query) / (np.linalg.norm(encoded_docs) * np.linalg.norm(encoded_query)))
        sims_with_ids = [(i, sims[i]) for i in range(len(sims))]
        sims_sorted = sorted(sims_with_ids, key=lambda x: (-x[1], x[0]))
        return [(self.doc_ids[pos_id], score) for pos_id, score in sims_sorted]


def perform_wilcoxon_test(expt1:str, expt2:str) -> Tuple[List[Tuple[float, float]] ,List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Performs the Wilcoxon signed-rank test for the two experiments (expt1 and expt2) and returns the p-value and statistic value.
    The test tells us if the difference in performance is statistically significant.

    Parameters:
    ------------
    expt1: str
        Name of the first experiment (must be same as the one used in main.py)
    expt2: str
        Name of the second experiment (must be same as the one used in main.py)

    Returns:
    ------------
    Tuple[List[Tuple[float, float]] ,List[Tuple[float, float]], List[Tuple[float, float]]]
        A tuple containing the p-value and statistic value of the Wilcoxon test
    """
    query_precisions_expt1 = json.load(open(out_dir_path / f"query_precisions_{expt1}.txt", 'r'))
    query_precisions_expt2 = json.load(open(out_dir_path / f"query_precisions_{expt2}.txt", 'r'))

    query_recalls_expt1 = json.load(open(out_dir_path / f"query_recalls_{expt1}.txt", 'r'))
    query_recalls_expt2 = json.load(open(out_dir_path / f"query_recalls_{expt2}.txt", 'r'))

    query_fscores_expt1 = json.load(open(out_dir_path / f"query_fscores_{expt1}.txt", 'r'))
    query_fscores_expt2 = json.load(open(out_dir_path / f"query_fscores_{expt2}.txt", 'r'))

    out_list_p = []
    out_list_r = []
    out_list_f = []

    for i in range(10):
        stat_p, p_value_p = wilcoxon(query_precisions_expt1[i], query_precisions_expt2[i], zero_method="zsplit")
        stat_r, p_value_r = wilcoxon(query_recalls_expt1[i], query_recalls_expt2[i], zero_method="zsplit")
        stat_f, p_value_f = wilcoxon(query_fscores_expt1[i], query_fscores_expt2[i], zero_method="zsplit")
        out_list_p.append((p_value_p, stat_p))
        out_list_r.append((p_value_r, stat_r))
        out_list_f.append((p_value_f, stat_f))

    return out_list_p, out_list_r, out_list_f


def expand_with_synonyms(qdoc: List[List[str]], vocabulary: set[str]) -> List[str]:
    outqdoc = []
    for sent in qdoc:
        outsent = list(sent)
        for word in sent:
            synsets = wordnet.synsets(word, pos=wordnet.NOUN)  
            if not synsets:
                continue
            for syn in synsets:
                for lemma in syn.lemmas():
                    l = lemma.name().replace('_', ' ').lower()
                    if l in vocabulary and l != word:
                        outsent.append(l)
        outqdoc.append(outsent)
    return outqdoc


def extract_thesaurus(thesaurus_path: Path, method_: str) -> None:

    def clean_and_split(phrase: str) -> List[str]:
        # Remove non-alphanumeric (except spaces), lowercase, split
        cleaned = re.sub(r'[^a-zA-Z\s]', '', phrase).lower()
        return [w for w in cleaned.split() if len(w) > 1]
    
    with open(thesaurus_path, 'r') as f:
        content = f.read()

    # Remove outer wrapping quotes from each line
    lines = []
    for line in content.splitlines():
        line = line.strip()
        # if Ice, Cloud and Land Elevation Satellite exists, replace the comma with empty space
        line = re.sub(r"Ice, Cloud and Land Elevation Satellite", "Ice Cloud and Land Elevation Satellite", line)
        # remove quotes
        line = re.sub(r'"', '', line)
        lines.append(line)

    cleaned = '\n'.join(lines)

    csv_file = pd.read_csv(
        StringIO(cleaned),
        doublequote=True,
        skipinitialspace=True,
    )

    thesaurus = {}
    for _, row in csv_file.iterrows():
        term = str(row['Key Descriptor']).lower()
        related = []
        
        rt = row['Relationship Type']
        if rt == 'Use':
            related += [row['Related Descriptor'].lower()]
        if rt in ['UF', 'RT']:
            related += [x.strip() for x in row['Related Descriptor'].lower().split(';')]
        
        if len(related) > 0:        
            thesaurus[term] = related
    
    lemmatizer = InflectionReduction()
    def lemmatize_phrase(phrase: str) -> List[str]:
        split_phrase = clean_and_split(phrase)
        doc = lemmatizer.reduce([split_phrase], method_ = method_)
        return doc[0]

    word_to_related = defaultdict(lambda: defaultdict(int))
    word_freq = defaultdict(int)

    for phrase, related_phrases in thesaurus.items():
        key_words = lemmatize_phrase(phrase)
        
        for rel_phrase in related_phrases:
            rel_words = lemmatize_phrase(rel_phrase)
            
            for kw in key_words:
                word_freq[kw] += 1
                for rw in rel_words:
                    if kw != rw:  # skip self-loops
                        word_to_related[kw][rw] += 1
    
    word_index = {}
    n_phrases = len(thesaurus)
    for word, related in word_to_related.items():
        idf = math.log(n_phrases / (1 + word_freq[word]))
        scored = [
            (w, count * idf)
            for w, count in related.items()
        ] # change this part (still better than raw counts, but not by much)
        word_index[word] = sorted(scored, key=lambda x: -x[1]) 

    with open(enclosing_dirpath / "models" / "comp_thesaurus.pkl", "wb") as f:
        pickle.dump(word_index, f)

    return None


def expand_query_with_thesaurus(query: List[List[str]], vocabulary:set[str], args) -> List[List[str]]:

    if not (enclosing_dirpath / "models" / "comp_thesaurus.pkl").exists():
        extract_thesaurus(thesaurus_path, args.inflect)
    with open(str(enclosing_dirpath / "models" / "comp_thesaurus.pkl"), "rb") as f:
        thesaurus = pickle.load(f)

    # Extract all scores from the thesaurus (flatten the list of tuples)
    thes_values = [score for related_list in thesaurus.values() for w, score in related_list]
    #print(sum(thes_values) / len(thes_values), max(thes_values), min(thes_values))
    word_blacklist = ["mercury", "venus", "earth", "mars", 'california', '']
    expanded_query = []
    for sentence in query:
        expanded_sentence = []
        for word in sentence:
            expanded_sentence.append(word)
            related = thesaurus.get(word, [])
            hits = [w for w, count in related if w in vocabulary and w not in expanded_sentence and w != word and count >= 10 and w not in word_blacklist][:2]
            expanded_sentence.extend(hits)
        expanded_query.append(expanded_sentence)

    return expanded_query


if __name__ == "__main__":
    expt1 = "baseline"
    expt2 = "lsa350"
    out_list_p, out_list_r, out_list_f = perform_wilcoxon_test(expt1, expt2)
    

    for i in range(10):
        print(f"Precision at {i+1}: p-value = {out_list_p[i][0]}, is statistically significant: {out_list_p[i][0] < 0.05}, statistic = {out_list_p[i][1]}")
        print(f"Recall at {i+1}: p-value = {out_list_r[i][0]}, is statistically significant: {out_list_r[i][0] < 0.05}, statistic = {out_list_r[i][1]}")
        print(f"F-score at {i+1}: p-value = {out_list_f[i][0]}, is statistically significant: {out_list_f[i][0] < 0.05}, statistic = {out_list_f[i][1]}")

    print(f"Wilcoxon test results for precision: {out_list_p}")
    print(f"Wilcoxon test results for recall: {out_list_r}")
    print(f"Wilcoxon test results for f-score: {out_list_f}")