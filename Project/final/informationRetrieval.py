from util import *

# Add your import statements here
import numpy as np
from textblob import TextBlob
from spellchecker import SpellChecker
from numpy.linalg import svd
import enchant
import sys
movies_dict = enchant.PyPWL("word.txt")

class InformationRetrieval():

	def __init__(self):
		self.index = None

	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""
		
		doc_inverted_list = {}
		for i in range(0,len(docs)):
			for j in range(0,len(docs[i])):
				for k in range(0,len(docs[i][j])):
					if docs[i][j][k] not in doc_inverted_list:
						doc_inverted_list[docs[i][j][k]] = [docIDs[i]]
					elif docIDs[i] not in doc_inverted_list[docs[i][j][k]] :
						doc_inverted_list[docs[i][j][k]].append(docIDs[i])

		self.index = (doc_inverted_list,docIDs,docs)


	def rank(self, queries, K = 400, w = 1.0):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""
		
		doc_IDs_ordered = []

		# Access elements from the index (assuming appropriate structure)
		doc_inverted_list, doc_no, docs = self.index
		n_docs = len(docs)
		Query = queries

		# Spell correction using TextBlob and movies_dict
		spell = SpellChecker()
		for query_i in range(0,len(Query)):
			for sent_j in range(0,len(Query[query_i])):
				# Check for misspelled words and correct if necessary
				misspelled = spell.unknown(Query[query_i][sent_j])
				for word_k in range(0,len(Query[query_i][sent_j])):
					if Query[query_i][sent_j][word_k] not in doc_inverted_list:

						b = TextBlob(Query[query_i][sent_j][word_k])
						b1 = Query[query_i][sent_j][word_k]
						Query[query_i][sent_j][word_k] = str(b.correct())
						b = str(b.correct())
						if b == b1 and b in misspelled:
							suggestions = movies_dict.suggest(b1)
							if len(suggestions) == 0:
								Query[query_i][sent_j][word_k] = b1 # Keep original word if no suggestions
							else:
								b = suggestions[0]
								Query[query_i][sent_j][word_k] = suggestions[0] # Use first suggestion
					# Handle words not in the inverted list and add them with empty document list
					if Query[query_i][sent_j][word_k] not in doc_inverted_list :
						doc_inverted_list[Query[query_i][sent_j][word_k]] = [0]

		# Doc-term matrix and word types initialization
		doc_term_matrix = np.zeros((n_docs, len(doc_inverted_list.keys())))
		words = list(doc_inverted_list.keys())

		# IDF calculation for all words
		idf = np.zeros((len(words), 1))
		for i in range(len(words)):
			idf[i] = np.log2(n_docs / (len(doc_inverted_list[words[i]])))

		# Find term frequencies and fill doc-term matrix
		for doc_i in range(len(docs)):
			for sent_j in range(len(docs[doc_i])):
				for word_k in range(0, len(docs[doc_i][sent_j])):
					word = docs[doc_i][sent_j][word_k]
					if word in doc_inverted_list:
						ind = words.index(word)

						# When not using weights
						doc_term_matrix[doc_i][ind] += 1 

						# Below part for using weights
						if sent_j == len(docs[doc_i])-1:
							doc_term_matrix[doc_i][ind] += w
						else:
							doc_term_matrix[doc_i][ind] += 1 

		# Convert TF to TF-IDF in doc-term matrix
		for doc_vec in range(doc_term_matrix.shape[0]):
			doc_term_matrix[doc_vec, :] = doc_term_matrix[doc_vec, :] * idf.T

		# Get TF values for query-term matrix
		query_term_matrix = np.zeros((len(Query), len(doc_inverted_list.keys())))
		for query_i in range(0, len(Query)):
			for sent_j in range(0, len(Query[query_i])):
				for word_k in range(0, len(Query[query_i][sent_j])):
					word = Query[query_i][sent_j][word_k]
					if word in doc_inverted_list:
						ind = list(doc_inverted_list.keys()).index(word)
						query_term_matrix[query_i][ind] += 1

		# Convert TF to TF-IDF in query-term matrix
		for doc_index in range(query_term_matrix.shape[0]):
			query_term_matrix[doc_index, :] = query_term_matrix[doc_index, :] * idf.T

		# Perform LSA
		U, S, V_T = svd(doc_term_matrix.T, full_matrices=False)

		Uk = U[:, :K]
		Sk = np.diag(S[:K])
		Vk_T = V_T[:K]

		# Update doc_term_matrix and query_term_matrix using LSA components
		doc_term_matrix = Vk_T.T @ Sk
		query_term_matrix = query_term_matrix @ Uk

		for query_i in range(len(query_term_matrix)):

			temp = []
			for word_vec_j in range(len(doc_term_matrix)):
				# Calculate cosine similarity between query vector and document vector
				cossim = np.dot(query_term_matrix[query_i, :], doc_term_matrix[word_vec_j, :]) / \
						((np.linalg.norm(query_term_matrix[query_i, :]) + 1e-6) *  # Add epsilon for numerical stability
						(np.linalg.norm(doc_term_matrix[word_vec_j, :]) + 1e-6))

				temp.append(cossim)

			# Sort document IDs by cosine similarity (descending order)
			sorted_doc_cosines = sorted(zip(temp, doc_no), reverse=True)

			# Extract document IDs from sorted cosine similarities
			doc_IDs_ordered.append([x for _, x in sorted_doc_cosines])
	
		return doc_IDs_ordered




