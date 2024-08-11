from util import *

# Add your import statements here

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
# nltk.download('wordnet')
# nltk.download('stopwords')


class InflectionReduction:

	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""

		reducedText = None

		#Fill in code here

		def wordnetpos(treebank_tag):

			if treebank_tag.startswith('J'):
				return wordnet.ADJ
			elif treebank_tag.startswith('V'):
				return wordnet.VERB
			elif treebank_tag.startswith('N'):
				return wordnet.NOUN
			elif treebank_tag.startswith('R'):
				return wordnet.ADV
			else:
				return wordnet.NOUN

		reducedText = []
		lemmatizer = WordNetLemmatizer()

		for i in range(len(text)):
			tagged=nltk.pos_tag(text[i])
			lemmas = []
			for j in range(len(text[i])):
				lemmas.append(lemmatizer.lemmatize(text[i][j], wordnetpos(tagged[j][1])))
			reducedText.append(lemmas)
		
		return reducedText


