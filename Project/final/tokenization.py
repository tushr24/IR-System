from util import *

# Add your import statements here
import nltk
import re
from nltk.tokenize import TreebankWordTokenizer




class Tokenization():

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

		tokenizedText = None

		#Fill in code here
		list1 = []
		for i in text:
			result = []
			a = [j.lower() for j in re.split('[ ]|[,]|[?]|[/]|[!]|[\"]|[\"]|[\']|[\']|[:]|[(]|[)]|[#]|[@]|[*]|[<]|[>]|[;]|[_]|[{]|[}]|[-]', i) if (len(j) > 0 and j != '')]
			list1.append(a)

		tokenizedText = list1
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

		tokenizedText = None

		#Fill in code here
		list1 = []
		
		tz = TreebankWordTokenizer()
		for i in text:
			a = tz.tokenize(i)
			list1.append(a)
		tokenizedText = list1
		return tokenizedText