import pickle

from util import *
import math
from collections import Counter, defaultdict
from additional_modules import GensimLSI, ESAWordRepn
import numpy as np
from pathlib import Path
from typing import List, Tuple

enclosing_dirpath = Path(__file__).resolve().parent
output_dir = enclosing_dirpath / "output"

class InformationRetrieval:

	def __init__(self):
		self.index = None
		self.doc_ids = []
		self.idf = {}
		self.doc_vectors = {}
		self.doc_norms = {}

	def _flatten(self, nested_sent_tokens:List[List[str]]):
		"""Flatten sentence-token structure into a lower-cased token list."""
		return [tok.lower() for sent in nested_sent_tokens for tok in sent if tok]
	
	def buildIndexESA(self, docs: List[List[List[str]]], docIDs: List[int]):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable. Uses wikipedia2vec's ESA

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""
		self.doc_ids = docIDs
		flattened_docs = [self._flatten(doc) for doc in docs]
		self.esa = ESAWordRepn(flattened_docs, docIDs)
		'''self.doc_vectors = {}
		for doc_ind, doc_id in enumerate(docIDs):
			doc_vector = self.esa.item2vec(flattened_docs[doc_ind])
			self.doc_vectors[doc_id] = doc_vector
		'''
	def rankESA(self, queries: List[List[List[str]]]):
		"""
		Rank the documents according to relevance for each query using wikipedia2vec's ESA

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""
		flattened_queries = [self._flatten(query) for query in queries]
		query_sims  = [self.esa.retrieve_docs(query) for query in flattened_queries] #List[List[Tuple[int, float]]]
		doc_IDs_ordered = [[tup[0] for tup in list2] for list2 in query_sims]

		return doc_IDs_ordered

	def buildIndexLSI(self, docs:List[List[List[str]]], docIDs:List[int], num_topics=550):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable while relying
		on LSI to provide similarity values between types.

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""
		flattened_docs = [self._flatten(doc) for doc in docs]
		self.indexer = GensimLSI(flattened_docs, num_topics, docIDs)
		self.sim_mat = self.indexer.index
		with open(output_dir / "lsi_topics.pkl", "wb") as f:
			pickle.dump(self.indexer.inspect_topics(), f)

	def rankLSI(self, queries:List[List[List[str]]]):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []
		scores = []
		for query in queries:
			query = self._flatten(query)
			sim_sorted = self.indexer.retrieve_docs(query)
			doc_IDs_ordered.append([sim_sorted[i][0] for i in range(len(sim_sorted))])
			scores.append([sim_sorted[i] for i in range(len(sim_sorted))])

		return doc_IDs_ordered
	

	def buildIndex(self, docs:List[List[List[str]]], docIDs:List[int]):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""

		index = {}
		self.doc_ids = list(docIDs)
		num_docs = len(docs)

		doc_term_counts = {}
		document_frequency = defaultdict(int)

		for doc, doc_id in zip(docs, docIDs):
			tf = Counter(self._flatten(doc))
			doc_term_counts[doc_id] = tf
			for term in tf:
				document_frequency[term] += 1

		# Smoothed IDF keeps values stable even for rare/common extremes.
		self.idf = {
			term: math.log((num_docs + 1.0) / (df + 1.0)) + 1.0
			for term, df in document_frequency.items()
		}

		self.doc_vectors = {}
		self.doc_norms = {}
		for doc_id in docIDs:
			tf = doc_term_counts[doc_id]
			vector = {}
			for term, freq in tf.items():
				# Log-scaled TF gives a stable baseline for Cranfield.
				vector[term] = (1.0 + math.log(freq)) * self.idf[term]

			norm = math.sqrt(sum(weight * weight for weight in vector.values()))
			self.doc_vectors[doc_id] = vector
			self.doc_norms[doc_id] = norm

		index["doc_vectors"] = self.doc_vectors
		index["idf"] = self.idf
		index["doc_ids"] = self.doc_ids

		self.index = index


	def rank(self, queries:List[List[List[str]]]):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []

		for query in queries:
			query_tf = Counter(self._flatten(query))
			query_vector = {}

			for term, freq in query_tf.items():
				if term in self.idf:
					query_vector[term] = (1.0 + math.log(freq)) * self.idf[term]

			query_norm = math.sqrt(sum(weight * weight for weight in query_vector.values()))

			if query_norm == 0.0:
				doc_IDs_ordered.append(list(self.doc_ids))
				continue

			scores = []
			for doc_id in self.doc_ids:
				doc_vector = self.doc_vectors[doc_id]
				doc_norm = self.doc_norms[doc_id]

				if doc_norm == 0.0:
					similarity = 0.0
				else:
					dot_product = 0.0
					for term, query_weight in query_vector.items():
						doc_weight = doc_vector.get(term)
						if doc_weight is not None:
							dot_product += query_weight * doc_weight
					similarity = dot_product / (query_norm * doc_norm)

				scores.append((doc_id, similarity))

			# Descending score, ascending doc_id for deterministic ordering.
			scores.sort(key=lambda item: (-item[1], item[0]))
			doc_IDs_ordered.append([doc_id for doc_id, _ in scores])
		
		return doc_IDs_ordered


