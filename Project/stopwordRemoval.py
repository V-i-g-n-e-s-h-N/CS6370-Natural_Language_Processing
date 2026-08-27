from nltk.corpus import stopwords as nltk_stopwords
import json
from pathlib import Path

def calculate_idf_dict():
	docs = dict()
	reduced_docs_path = Path(__file__).resolve().parent / 'output' / 'reduced_docs.txt'
	with open(reduced_docs_path, 'r') as file:
		loaded_file = json.load(file)
		for k, v in enumerate(loaded_file): # iterating over documents
			word_rep = set() #tracks repeated words
			for _, l in enumerate(v): #iterating over sentences in documents
				for word in l: #iterating over words in sentences
					if word not in docs.keys(): # first occurence of a word
						docs[word] = 1
						word_rep.add(word) 
					elif word not in word_rep: # if word has occured in a previous document, but not in current one
						docs[word] += 1 
						word_rep.add(word)

	idf_docs = {k : len(loaded_file)/v for k,v in docs.items()}

	return idf_docs, len(loaded_file)


def data_driven_stop(idf, N):
	stop_words = [k for k in idf.keys() if idf[k] < 0.001 * N] #idf thresholding
	stops = set(stop_words)
	return stops

class StopwordRemoval:

    def __init__(self, is_data_driven=True):
        self.is_data_driven = is_data_driven
        if self.is_data_driven:
            idf_dict, N = calculate_idf_dict()
            self._stopwords = data_driven_stop(idf_dict, N)
        else:
            self._stopwords = set(nltk_stopwords.words("english"))
        
        stop_word_path = Path(__file__).resolve().parent / 'output' / 'stopwords.txt'
        with open(stop_word_path, 'w') as file:
            json.dump(list(self._stopwords), file)

    def fromList(self, text):
        return [[tok for tok in sent if tok.lower() not in self._stopwords] for sent in text]
	
    
