from util import *
import math

class Evaluation:

	def _relevant_set(self, qrels, query_id):
		"""Binary relevance set using assignment rule: positions 1..4 are relevant."""
		relevant = set()
		for row in qrels:
			if int(row["query_num"]) == int(query_id):
				relevance = int(row["position"])
				if 1 <= relevance <= 4:
					relevant.add(int(row["id"]))
		return relevant

	def _graded_relevance(self, qrels, query_id):
		graded = {}
		for row in qrels:
			if int(row["query_num"]) == int(query_id):
				doc_id = int(row["id"])
				position = int(row["position"])
				if 1 <= position <= 4:
					graded[doc_id] = max(graded.get(doc_id, 0), 5 - position)
		return graded

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""Return Precision@k for a single query."""

		precision = 0.0
		if k <= 0:
			return precision
		top_k = query_doc_IDs_ordered[:k]
		if not top_k:
			return precision
		hits = sum(1 for doc_id in top_k if doc_id in true_doc_IDs)
		precision = hits / float(k)

		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""Return mean Precision@k across all queries."""
		meanPrecision = 0.0
		if not query_ids:
			return meanPrecision
		values = []
		for ranked_docs, query_id in zip(doc_IDs_ordered, query_ids):
			true_doc_IDs = self._relevant_set(qrels, query_id)
			values.append(self.queryPrecision(ranked_docs, query_id, true_doc_IDs, k))
		meanPrecision = sum(values) / float(len(values))

		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""Return Recall@k for a single query."""
		recall = 0.0
		if not true_doc_IDs:
			return recall
		top_k = query_doc_IDs_ordered[:k]
		hits = sum(1 for doc_id in top_k if doc_id in true_doc_IDs)
		recall = hits / float(len(true_doc_IDs))

		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""Return mean Recall@k across all queries."""
		meanRecall = 0.0
		if not query_ids:
			return meanRecall
		values = []
		for ranked_docs, query_id in zip(doc_IDs_ordered, query_ids):
			true_doc_IDs = self._relevant_set(qrels, query_id)
			values.append(self.queryRecall(ranked_docs, query_id, true_doc_IDs, k))
		meanRecall = sum(values) / float(len(values))

		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""Return F0.5@k for a single query."""
		fscore = 0.0
		beta = 0.5
		precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		denominator = (beta * beta * precision) + recall
		if denominator == 0.0:
			return fscore
		fscore = ((1 + beta * beta) * precision * recall) / denominator

		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""Return mean F0.5@k across all queries."""
		meanFscore = 0.0
		if not query_ids:
			return meanFscore
		values = []
		for ranked_docs, query_id in zip(doc_IDs_ordered, query_ids):
			true_doc_IDs = self._relevant_set(qrels, query_id)
			values.append(self.queryFscore(ranked_docs, query_id, true_doc_IDs, k))
		meanFscore = sum(values) / float(len(values))

		return meanFscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""Return nDCG@k for a single query using graded relevance."""
		nDCG = 0.0
		# Here true_doc_IDs is a graded relevance map: {doc_id: grade}.
		graded = true_doc_IDs if isinstance(true_doc_IDs, dict) else {}
		top_k = query_doc_IDs_ordered[:k]

		dcg = 0.0
		for i, doc_id in enumerate(top_k, start=1):
			rel = graded.get(doc_id, 0)
			if i == 1:
				dcg += rel
			else:
				dcg += rel / math.log2(i)

		ideal_rels = sorted(graded.values(), reverse=True)[:k]
		idcg = 0.0
		for i, rel in enumerate(ideal_rels, start=1):
			if i == 1:
				idcg += rel
			else:
				idcg += rel / math.log2(i)

		if idcg == 0.0:
			return nDCG
		nDCG = dcg / idcg

		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""Return mean nDCG@k across all queries."""
		meanNDCG = 0.0
		if not query_ids:
			return meanNDCG
		values = []
		for ranked_docs, query_id in zip(doc_IDs_ordered, query_ids):
			graded = self._graded_relevance(qrels, query_id)
			values.append(self.queryNDCG(ranked_docs, query_id, graded, k))
		meanNDCG = sum(values) / float(len(values))

		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""Return AP@k for a single query."""
		avgPrecision = 0.0
		if not true_doc_IDs:
			return avgPrecision

		num_hits = 0
		precision_sum = 0.0
		for rank, doc_id in enumerate(query_doc_IDs_ordered[:k], start=1):
			if doc_id in true_doc_IDs:
				num_hits += 1
				precision_sum += num_hits / float(rank)

		denominator = min(len(true_doc_IDs), k)
		if denominator == 0:
			return avgPrecision
		avgPrecision = precision_sum / float(denominator)

		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""Return MAP@k across all queries."""
		meanAveragePrecision = 0.0
		if not query_ids:
			return meanAveragePrecision
		values = []
		for ranked_docs, query_id in zip(doc_IDs_ordered, query_ids):
			true_doc_IDs = self._relevant_set(q_rels, query_id)
			values.append(self.queryAveragePrecision(ranked_docs, query_id, true_doc_IDs, k))
		meanAveragePrecision = sum(values) / float(len(values))

		return meanAveragePrecision



	def queryReciprocalRank(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""Return reciprocal rank@k for a single query."""

		reciprocalRank = 0.0
		for rank, doc_id in enumerate(query_doc_IDs_ordered[:k], start=1):
			if doc_id in true_doc_IDs:
				reciprocalRank = 1.0 / float(rank)
				break

		return reciprocalRank


	def meanReciprocalRank(self, doc_IDs_ordered, query_ids, qrels, k):
		"""Return MRR@k across all queries."""

		meanReciprocalRank = 0.0
		if not query_ids:
			return meanReciprocalRank
		values = []
		for ranked_docs, query_id in zip(doc_IDs_ordered, query_ids):
			true_doc_IDs = self._relevant_set(qrels, query_id)
			values.append(self.queryReciprocalRank(ranked_docs, query_id, true_doc_IDs, k))
		meanReciprocalRank = sum(values) / float(len(values))

		return meanReciprocalRank
