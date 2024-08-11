from util import *

# Add your import statements here
import nltk
from nltk.corpus import stopwords


class StopwordRemoval():

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

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

		stopwordRemovedText = None

		#Fill in code here
		
		checklist = stopwords.words('english')
		for i in range(0,len(text)):
			a = []
			for j in text[i]:
				if j not in checklist:
					a.append(j)
			text[i] = a
		
		stopwordRemovedText = text
		return stopwordRemovedText




	