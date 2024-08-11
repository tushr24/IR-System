from util import *

# Add your import statements here
import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
import nltk

from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
from nltk import tokenize


class SentenceSegmentation():

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

		text = text.strip()
		sentences = re.split('[.]|[!]|[?]|[;]',text)
		segmentedText = [s.strip() for s in sentences if (len(s)>=2)]

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
			A list of strings where each strin is a single sentence
		"""

		segmentedText = None

		
		segmentedText = tokenize.sent_tokenize(text)

		
		return segmentedText