# Add your import statements here
import nltk

#Centralized downloading of NLTK resources to avoid multiple calls across files
def download_nltk_resources():
	resources = [
		'punkt', 
		'punkt_tab', 
		'stopwords', 
		'wordnet', 
		'averaged_perceptron_tagger_eng'
	]
	for res in resources:
		nltk.download(res, quiet=True)

#Run the download when util is imported
download_nltk_resources()

# Add any utility functions here

def get_wordnet_pos(tag):
	"""
	Helper method to map NLTK's Treebank POS tags to WordNet POS tags.
	This is commonly used in WordNetLemmatizer across different modules.
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