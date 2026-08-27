import nltk
import spacy
from nltk.stem import PorterStemmer
import numpy as np
import re
from pathlib import Path
from symspellpy import SymSpell
from typing import List

current_path = Path(__file__).resolve()
parent_dir = current_path.parent
output_dir = parent_dir / "output"
model_path = parent_dir / "models" / "Model_for_ESA"
freq_dict = parent_dir / "models" / "freq_dict" / "frequency_dictionary_en_82_765.txt"

sym_spell = SymSpell(max_dictionary_edit_distance=2)
sym_spell.load_dictionary(str(freq_dict), 0, 1)

def correct_word(word: str) -> str:
    suggestions = sym_spell.lookup(word, verbosity=0, max_edit_distance=2)
    return suggestions[0].term if suggestions else word


nlp = spacy.load("en_core_web_sm")


class InflectionReduction:

	def porterStemmer(self, text: List[List[str]]) -> List[List[str]]:
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

		reducedText = None

		# Fill in code here
		ps = PorterStemmer()
		reducedText = [[ps.stem(correct_word(word)) for word in sent] for sent in text]	# text is a list of sentences; sentence is a list of words

		return reducedText



	def spacyLemmatizer(self, text: List[List[str]]) -> List[List[str]]:
		"""
        Inflection Reduction using spaCy Lemmatizer
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
		for sent in text:
			sent = [correct_word(re.sub(r'[^a-zA-Z0-9-]', '', word)) for word in sent] #removing non-alphanumeric characters except hyphen
			lem_sent_join = nlp(" ".join(sent))
			lemmatized_sentence = [re.sub(r'[^a-zA-Z0-9]', '', token.lemma_) for token in lem_sent_join] # we needed hyphen earlier to facilitate lemmatization of hyphenated words
					
			reducedText.append(lemmatized_sentence)
		

		return reducedText
	
	def reduce(self, text: List[List[str]], method_ : str) -> List[List[str]]:
		"""
		Wrapper function for inflection reduction.
		Also added the raw option 
		"""
		
		reducedText = None

		# Fill in code here
		if method_ == 'lemma':
			reducedText = self.spacyLemmatizer(text)
		elif method_ == 'stem':
			reducedText = self.porterStemmer(text)
		else:
			reducedText = text
		return reducedText
