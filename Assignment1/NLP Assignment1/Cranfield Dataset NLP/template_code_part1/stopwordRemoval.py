from util import *
# Add your import statements here
import nltk
from nltk.corpus import stopwords
from collections import Counter

# --- DEFENSIVE PROGRAMMING ---
nltk.download('stopwords', quiet=True)

class StopwordRemoval():

	def __init__(self):
		# Load NLTK stopwords into a set for extremely fast O(1) lookups
		self.nltk_stopwords = set(stopwords.words('english'))

	def fromList(self, text):
		"""
		Stopword removal using the curated NLTK list.

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""
		stopwordRemovedText = []
		for sentence in text:
			# Keep the word only if its lowercase form is not in the NLTK stopword set
			filtered_sentence = [word for word in sentence if word.lower() not in self.nltk_stopwords]
			stopwordRemovedText.append(filtered_sentence)
		return stopwordRemovedText

	def dataDriven(self, text, top_k=50):
		"""
		Stopword removal using a bottom-up, data-driven approach based on Term Frequency.

		Returns the filtered text AND the generated custom stopword set so you can compare them.
		"""
		# Flatten the list of lists into a single list of all words in the corpus
		all_words = [word.lower() for sentence in text for word in sentence]

		# Count the frequency of every word
		word_counts = Counter(all_words)

		# Extract the 'top_k' most frequent words to act as our custom stopword list
		custom_stopwords = set([word for word, count in word_counts.most_common(top_k)])

		stopwordRemovedText = []
		for sentence in text:
			filtered_sentence = [word for word in sentence if word.lower() not in custom_stopwords]
			stopwordRemovedText.append(filtered_sentence)

		return stopwordRemovedText, custom_stopwords




