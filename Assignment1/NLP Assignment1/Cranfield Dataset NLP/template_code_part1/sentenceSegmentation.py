from util import *

# Add your import statements here
import re
import nltk
import spacy
from nltk.tokenize import sent_tokenize, PunktTokenizer
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# NOTE: This code requires the spaCy English model. If not installed, please run: python -m spacy download en_core_web_sm

class SentenceSegmentation():

	def __init__(self):
		# Load spaCy model (students may use this if needed)
		self.nlp = spacy.load("en_core_web_sm")

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = None
		#Our approach here is to have a top-down set of rules to detect patterns that are followed by sentences predominantly
		#I am only considering the following situation - a punctuation followed by a space, that is not a part of a number or a part of an abbreviation, though it may end an abbreviation
		#regex is used for this pattern matching
		pattern = r'(?<=[.!?])\s+(?=[A-Z, 1-9])'
		segmentedText = re.split(pattern, text)
		return segmentedText


	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = None
		# We now use a pretrained tokenizer, in order to load the trained weights, we do not use PunktTaokenizer, but its wrapper - sent_tokenize
		text = text.strip()
		segmentedText = sent_tokenize(text)
		return segmentedText

	def spacySegmenter(self, text):
		"""
		Sentence Segmentation using spaCy

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""
		segmentedText = None
		#We use the pretrained sentence segmentation model of spacy, that was trained for general purpose English text found on the web
		segmentedTextCorpus = self.nlp(text)
		#We convert the extracted sentences to a list, and since we require string outputs, we use the .text method
		segmentedText = [sentence.text for sentence in segmentedTextCorpus.sents]
		return segmentedText
