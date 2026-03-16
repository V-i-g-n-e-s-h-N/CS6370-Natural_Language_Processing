from util import *

# Add your import statements here
# (Students may import required libraries such as nltk, spacy, re, etc.)
import re
import spacy
from nltk.tokenize import TreebankWordTokenizer

class Tokenization():

	def __init__(self):
        #Load spaCy model for the spaCy tokenizer
		self.nlp = spacy.load("en_core_web_sm")
        #Initialize the Penn Treebank Tokenizer
		self.ptb_tokenizer = TreebankWordTokenizer()

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []
		#Append every string of alphanumeric characters as a word to the list of tokens, and stop for every whitespace character. 
		#If a non-whitespace non-alphanumeric chacrater is found, then it is separately tokenized
		pattern = r'\w+|[^\w\s]'
		for sentence in text:
			tokens = re.findall(pattern, sentence)
			tokenizedText.append(tokens)
		return tokenizedText


	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []
        # This tokenizer handles clitics and punctuation intelligently based on PTB rules
		for sentence in text:
			tokens = self.ptb_tokenizer.tokenize(sentence)
			tokenizedText.append(tokens)
		return tokenizedText

	def spacyTokenizer(self, text):
		"""
		Tokenization using spaCy

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []
        # spaCy uses a hybrid approach (whitespace splitting + exception dictionaries + affix rules)
		for sentence in text:
			doc = self.nlp(sentence)
            # Extract the string representation (.text) of each token
			tokens = [token.text for token in doc]
			tokenizedText.append(tokens)
		return tokenizedText
