from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from informationRetrieval import InformationRetrieval
from evaluation import Evaluation
from additional_modules import expand_with_synonyms, expand_query_with_thesaurus, extract_thesaurus

from sys import version_info
import argparse
import json
import pickle
import matplotlib.pyplot as plt
import os
from pathlib import Path

enclosing_dirpath = Path(__file__).resolve().parent
thesaurus_path = enclosing_dirpath / "models" / "NASA_thesaurus_CSV.txt"

# Input compatibility for Python 2 and Python 3
if version_info.major == 3:
    pass
elif version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass
else:
    print("Unknown python version - input function not safe")

# find representative queries using LSI - or find out worst performing queries
# find how they perform (how many relevant documents are missed)

class SearchEngine:

    def __init__(self, args):
        self.args = args

        # Create output folder if it does not exist.
        if not os.path.exists(self.args.out_folder):
            os.makedirs(self.args.out_folder)

        self.tokenizer = Tokenization()
        self.sentenceSegmenter = SentenceSegmentation()
        self.inflectionReducer = InflectionReduction()
        self.stopwordRemover = StopwordRemoval(is_data_driven=self.args.datadriven_stop)

        self.informationRetriever = InformationRetrieval()
        self.evaluator = Evaluation()

    def segmentSentences(self, text):
        if self.args.segmenter == "naive":
            return self.sentenceSegmenter.naive(text)
        elif self.args.segmenter == "punkt":
            return self.sentenceSegmenter.punkt(text)

    def tokenize(self, text):
        if self.args.tokenizer == "naive":
            return self.tokenizer.naive(text)
        elif self.args.tokenizer == "ptb":
            return self.tokenizer.pennTreeBank(text)

    def reduceInflection(self, text):
        return self.inflectionReducer.reduce(text, method_=self.args.inflect)

    def removeStopwords(self, text):
        return self.stopwordRemover.fromList(text)

    def preprocessQueries(self, queries):
        segmentedQueries = []
        for query in queries:
            segmentedQuery = self.segmentSentences(query)
            segmentedQueries.append(segmentedQuery)

        json.dump(segmentedQueries, open(os.path.join(self.args.out_folder, "segmented_queries.txt"), 'w'))

        tokenizedQueries = []
        for query in segmentedQueries:
            tokenizedQuery = self.tokenize(query)
            tokenizedQueries.append(tokenizedQuery)

        json.dump(tokenizedQueries, open(os.path.join(self.args.out_folder, "tokenized_queries.txt"), 'w'))

        vocab_path = enclosing_dirpath / "output"/ "vocabulary.pkl"
        
        with open(vocab_path, "rb") as f:
            vocabulary = pickle.load(f)

        reducedQueries = []
        for query in tokenizedQueries:
            if self.args.expand == 'thesaurus':
                query = expand_query_with_thesaurus(query, vocabulary, self.args)
            elif self.args.expand == 'synonyms':
                query = expand_with_synonyms(query, vocabulary)
            # else: no expansion
            reducedQuery = self.reduceInflection(query)
            reducedQueries.append(reducedQuery)

        json.dump(reducedQueries, open(os.path.join(self.args.out_folder, "reduced_queries.txt"), 'w'))

        stopwordRemovedQueries = []
        for query in reducedQueries:
            stopwordRemovedQuery = self.removeStopwords(query)
            stopwordRemovedQueries.append(stopwordRemovedQuery)

        json.dump(stopwordRemovedQueries, open(os.path.join(self.args.out_folder, "stopword_removed_queries.txt"), 'w'))

        return stopwordRemovedQueries

    def preprocessDocs(self, docs):
        segmentedDocs = []
        for doc in docs:
            segmentedDoc = self.segmentSentences(doc)
            segmentedDocs.append(segmentedDoc)

        json.dump(segmentedDocs, open(os.path.join(self.args.out_folder, "segmented_docs.txt"), 'w'))

        tokenizedDocs = []
        for doc in segmentedDocs:
            tokenizedDoc = self.tokenize(doc)
            tokenizedDocs.append(tokenizedDoc)

        json.dump(tokenizedDocs, open(os.path.join(self.args.out_folder, "tokenized_docs.txt"), 'w'))

        reducedDocs = []
        for doc in tokenizedDocs:
            reducedDoc = self.reduceInflection(doc)
            reducedDocs.append(reducedDoc)

        json.dump(reducedDocs, open(os.path.join(self.args.out_folder, "reduced_docs.txt"), 'w'))

        stopwordRemovedDocs = []
        for doc in reducedDocs:
            stopwordRemovedDoc = self.removeStopwords(doc)
            stopwordRemovedDocs.append(stopwordRemovedDoc)

        json.dump(stopwordRemovedDocs, open(os.path.join(self.args.out_folder, "stopword_removed_docs.txt"), 'w'))

        vocabulary = set(word for doc in stopwordRemovedDocs for sent in doc for word in sent)
        vocab_path = enclosing_dirpath / "output"/ "vocabulary.pkl"
        if not vocab_path.exists():
            pickle.dump(vocabulary, open(vocab_path, 'wb'))

        return stopwordRemovedDocs

    def evaluateDataset(self):

        docs_json = json.load(open(os.path.join(args.dataset, "cran_docs.json"), 'r'))[:]
        doc_ids = [item["id"] for item in docs_json]
        docs = [item["body"] for item in docs_json]

        processedDocs = self.preprocessDocs(docs)

        queries_json = json.load(open(os.path.join(args.dataset, "cran_queries.json"), 'r'))[:]
        query_ids = [item["query number"] for item in queries_json]
        queries = [item["query"] for item in queries_json]

        processedQueries = self.preprocessQueries(queries)


        if args.algo == 'lsa':
            self.informationRetriever.buildIndexLSI(processedDocs, doc_ids)
            doc_IDs_ordered = self.informationRetriever.rankLSI(processedQueries)
        elif args.algo == 'esa':
            self.informationRetriever.buildIndexESA(processedDocs, doc_ids)
            doc_IDs_ordered = self.informationRetriever.rankESA(processedQueries)
        else:
            self.informationRetriever.buildIndex(processedDocs, doc_ids)
            doc_IDs_ordered = self.informationRetriever.rank(processedQueries)

        qrels = json.load(open(os.path.join(args.dataset, "cran_qrels.json"), 'r'))[:]

        precisions, recalls, fscores, MAPs, nDCGs, MRRs = [], [], [], [], [], []

        for k in range(1, 11):

            precision = self.evaluator.meanPrecision(doc_IDs_ordered, query_ids, qrels, k)
            recall = self.evaluator.meanRecall(doc_IDs_ordered, query_ids, qrels, k)
            fscore = self.evaluator.meanFscore(doc_IDs_ordered, query_ids, qrels, k)

            precisions.append(precision)
            recalls.append(recall)
            fscores.append(fscore)

            print(f"Precision, Recall, F-score @ {k}: {precision}, {recall}, {fscore}")

            MAP = self.evaluator.meanAveragePrecision(doc_IDs_ordered, query_ids, qrels, k)
            nDCG = self.evaluator.meanNDCG(doc_IDs_ordered, query_ids, qrels, k)
            MRR = self.evaluator.meanReciprocalRank(doc_IDs_ordered, query_ids, qrels, k)

            MAPs.append(MAP)
            nDCGs.append(nDCG)
            MRRs.append(MRR)

            print(f"MAP, nDCG, MRR @ {k}: {MAP}, {nDCG}, {MRR}")

        query_precisions = [[self.evaluator.queryPrecision(doc_IDs_ordered[i], query_ids[i], self.evaluator._relevant_set(qrels, query_ids[i]), j) for i in range(len(query_ids))] for j in range(1, 11)]
        query_recalls = [[self.evaluator.queryRecall(doc_IDs_ordered[i], query_ids[i], self.evaluator._relevant_set(qrels, query_ids[i]), j) for i in range(len(query_ids))] for j in range(1, 11)]
        query_fscores = [[self.evaluator.queryFscore(doc_IDs_ordered[i], query_ids[i], self.evaluator._relevant_set(qrels, query_ids[i]), j) for i in range(len(query_ids))] for j in range(1, 11)]

        json.dump(query_precisions, open(os.path.join(self.args.out_folder, f"query_precisions_{self.args.expt_name}.txt"), 'w'))
        json.dump(query_recalls, open(os.path.join(self.args.out_folder, f"query_recalls_{self.args.expt_name}.txt"), 'w'))
        json.dump(query_fscores, open(os.path.join(self.args.out_folder, f"query_fscores_{self.args.expt_name}.txt"), 'w'))

        worst_10_queries_prec = [sorted(zip(query_ids, query_precisions[j]), key=lambda x: x[1])[:10] for j in range(10)]
        worst_10_queries_rec = [sorted(zip(query_ids, query_recalls[j]), key=lambda x: x[1])[:10] for j in range(10)]
        worst_10_queries_fsc = [sorted(zip(query_ids, query_fscores[j]), key=lambda x: x[1])[:10] for j in range(10)]

        for i in range(10):
            p_query = query_ids.index(worst_10_queries_prec[0][i][0])
            r_query = query_ids.index(worst_10_queries_rec[0][i][0])
            f_query = query_ids.index(worst_10_queries_fsc[0][i][0])

            print(f"Worst query {i+1} by precision@1: Query ID: {query_ids[p_query]}, Query {processedQueries[p_query]}, Precision: {worst_10_queries_prec[0][i][1]}, Retrieved Docs: {set(doc_IDs_ordered[p_query][:10])}")
            print(f"Worst query {i+1} by recall@1: Query ID: {query_ids[r_query]}, Query {processedQueries[r_query]}, Recall: {worst_10_queries_rec[0][i][1]}, Retrieved Docs: {set(doc_IDs_ordered[r_query][:10])}")
            print(f"Worst query {i+1} by f-score@1: Query ID: {query_ids[f_query]}, Query {processedQueries[f_query]}, F-Score: {worst_10_queries_fsc[0][i][1]}, Retrieved Docs: {set(doc_IDs_ordered[f_query][:10])}")

        # Plot
        plt.plot(range(1, 11), precisions, label="Precision")
        plt.plot(range(1, 11), recalls, label="Recall")
        plt.plot(range(1, 11), fscores, label="F-Score")
        plt.plot(range(1, 11), MAPs, label="MAP")
        plt.plot(range(1, 11), nDCGs, label="nDCG")
        plt.plot(range(1, 11), MRRs, label="MRR")

        plt.legend()
        plt.title("Evaluation Metrics - Cranfield Dataset")
        plt.xlabel("k")
        plt.ylabel("Score")
        plt.savefig(os.path.join(self.args.out_folder, "eval_plot_stem.png"))

    def handleCustomQuery(self):

        print("Enter query below")
        query = input()

        processedQuery = self.preprocessQueries([query])[0]

        docs_json = json.load(open(os.path.join(args.dataset, "cran_docs.json"), 'r'))[:]
        doc_ids = [item["id"] for item in docs_json]
        docs = [item["body"] for item in docs_json]

        processedDocs = self.preprocessDocs(docs)

        self.informationRetriever.buildIndex(processedDocs, doc_ids)
        doc_IDs_ordered = self.informationRetriever.rank([processedQuery])[0]

        print("\nTop five document IDs : ")
        for id_ in doc_IDs_ordered[:5]:
            print(id_)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='main.py')

    parser.add_argument('-dataset', default="cranfield/")
    parser.add_argument('-out_folder', default="output/")
    parser.add_argument('-segmenter', default="punkt")
    parser.add_argument('-tokenizer', default="ptb")
    parser.add_argument('-custom', action="store_true")
    parser.add_argument('-inflect', choices=["raw", "stem", "lemma"], default="stem")
    parser.add_argument('-algo', choices=['vanilla', 'lsa', 'esa'], default='vanilla')
    parser.add_argument('-expt_name', default="baseline")
    parser.add_argument('-datadriven_stop', action="store_true")
    parser.add_argument('-expand', choices=['none', 'thesaurus', 'synonyms'], default='none')

    args = parser.parse_args()

    word_pairs_path = enclosing_dirpath / "models" / "comp_thesaurus.pkl"
    if not word_pairs_path.exists():
        extract_thesaurus(thesaurus_path, args.inflect)

    searchEngine = SearchEngine(args)

    if args.custom:
        searchEngine.handleCustomQuery()
    else:
        searchEngine.evaluateDataset()