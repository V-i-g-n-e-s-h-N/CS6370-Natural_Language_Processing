from util import *

# Add your import statements here
# (Students may import required libraries such as nltk, WordNetLemmatizer, PorterStemmer, etc.)
import nltk
from nltk import pos_tag
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('wordnet', quiet=True)

class InflectionReduction:

	def __init__(self):
        # Initialize the tools once to save computational overhead
		self.stemmer = PorterStemmer()
		self.lemmatizer = WordNetLemmatizer()

	def get_wordnet_pos(self, tag):
		"""
		Helper method to map NLTK's Treebank POS tags to WordNet POS tags.
		"""
		if tag.startswith('J'):
			return 'a' #Adjective
		elif tag.startswith('V'):
			return 'v' #Verb
		elif tag.startswith('N'):
			return 'n' #Noun
		elif tag.startswith('R'):
			return 'r' #Adverb
		else:
			return 'n' #Default to noun

	def porterStemmer(self, text):
		"""
		Inflection Reduction using Porter Stemmer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed tokens representing a sentence
		"""

		reducedText = []
		for sentence in text:
            #Apply the stemmer to each token in the sentence
			stemmed_sentence = [self.stemmer.stem(token) for token in sentence]
			reducedText.append(stemmed_sentence)
		return reducedText



	def wordnetLemmatizer(self, text):
		"""
		Inflection Reduction using WordNet Lemmatizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			lemmatized tokens representing a sentence
		"""

		reducedText = []
		for sentence in text:
            #Apply the lemmatizer to each token. 
            #Without explicit POS tags passed in, WordNetLemmatizer defaults to treating words as nouns. 
			#So to improve the quality of the lemmatization, we also use pos tagging
			lemmatized_sentence = []
			tagged_tokens = pos_tag(sentence)
			for word, tag in tagged_tokens:
				wn_tag = self.get_wordnet_pos(tag)
				lemmatized_sentence.append(self.lemmatizer.lemmatize(word, wn_tag))
			reducedText.append(lemmatized_sentence)
		return reducedText


	def reduce(self, text):
		"""
		Wrapper function for inflection reduction.
		Students may choose which method to call
		or extend this function to support both options.
		"""

		#Use the lemmatizer as the standard default function
		reducedText = self.wordnetLemmatizer(text)
		return reducedText
