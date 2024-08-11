from util import *

# Add your import statements here


class Evaluation():
	
	def true_doc_id_list(self, query_id, qrels):

		true_doc_ids = []
		# Iterate through each relevance judgment
		for judgement in qrels:
			# Check if the current judgment belongs to the specified query
			if judgement['query_num'] == str(query_id):  # Convert query_id to string for comparison
				true_doc_ids.append(int(judgement['id']))

		return true_doc_ids
		

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		# Calculate the intersection size between retrieved and relevant documents at rank k
		intersection_size = len(set(query_doc_IDs_ordered[:k]) & set(true_doc_IDs))

		# Calculate precision at rank k
		precision = intersection_size / k

		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		# Calculate mean precision at rank k
		meanPrecision = 0
		for query_id in query_ids:
			# Retrieve relevant documents for the current query
			true_doc_ids = self.true_doc_id_list(query_id, qrels)
			
			# Calculate precision for the current query
			order_doc_ID = doc_IDs_ordered[query_ids.index(query_id)]
			precision = self.queryPrecision(order_doc_ID, query_id, true_doc_ids, k)
			
			# Accumulate precision for all queries
			meanPrecision += precision

		# Calculate the overall mean precision
		meanPrecision /= len(query_ids)

		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		# Calculate the intersection size between retrieved and relevant documents at rank k
		intersection_size = len(set(query_doc_IDs_ordered[:k]) & set(true_doc_IDs))

		# Calculate recall at rank k, 0 if len(true_doc_IDs) <= 0
		recall = intersection_size / len(true_doc_IDs) if len(true_doc_IDs) > 0 else 0

		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		# Calculate mean recall at rank k
		meanRecall = 0
		for query_id in query_ids:
			# Retrieve relevant documents for the current query
			true_doc_ids = self.true_doc_id_list(query_id, qrels)
			
			# Calculate recall for the current query
			recall = self.queryRecall(doc_IDs_ordered[query_ids.index(query_id)], query_id, true_doc_ids, k)
			
			# Accumulate recall for all queries
			meanRecall += recall

		# Calculate the overall mean recall
		meanRecall /= len(query_ids)

		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		# Calculate precision and recall for the current query
		precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)

		# Handle division by zero for F-score calculation
		if precision == 0 and recall == 0:
			fscore = 0
		else:
			# Calculate F1-score (harmonic mean of precision and recall)
			fscore = (2 * precision * recall) / (precision + recall)

		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		# Calculate mean F-score at rank k
		meanFscore = 0
		for query_id in query_ids:
			# Retrieve relevant documents for the current query
			true_doc_ids = self.true_doc_id_list(query_id, qrels)
			
			# Calculate F-score for the current query
			f_score = self.queryFscore(doc_IDs_ordered[query_ids.index(query_id)], query_id, true_doc_ids, k)
			
			# Accumulate F-score for all queries
			meanFscore += f_score

		# Calculate the overall mean F-score
		meanFscore /= len(query_ids)


		return meanFscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

		# Relevance scores (initialization with zeros)
		rel = np.zeros((len(query_doc_IDs_ordered), 1))
	
		true_doc_IDs["position"] = 5-true_doc_IDs["position"] 

		# Sort documents by decreasing position (relevance or rank)
		true_doc_ids_sorted = true_doc_IDs.sort_values("position", ascending=False)

		# Ideal DCG (discounted sum of ideal positions)
		iDCG = true_doc_ids_sorted.iloc[0]["position"]
		for i in range(1, min(k,len(true_doc_IDs))):
			iDCG += true_doc_ids_sorted.iloc[i]["position"] / np.log2(i+1)

		# List of document IDs from the true doc IDs subset (for efficiency)
		t_docs = list(map(int, true_doc_IDs["id"]))

		# Calculate relevance scores for retrieved documents
		for i in range(k):
			if query_doc_IDs_ordered[i] in t_docs:
				rel[i] = true_doc_IDs[true_doc_IDs["id"] == str(query_doc_IDs_ordered[i])].iloc[0]["position"]

		# Discounted Cumulative Gain (DCG)
		DCG = 0
		for i in range(k):
			DCG += rel[i] / np.log2(i + 2)  

		# Normalized Discounted Cumulative Gain (nDCG)
		nDCG = DCG / iDCG

		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		count = 0
		qrels_df = pd.DataFrame(qrels)
		for i, query_id in enumerate(query_ids):
			query_doc_ids_ordered = doc_IDs_ordered[i]

			# Filter true doc IDs for the current query
			true_doc_ids = qrels_df[["position", "id"]][qrels_df["query_num"] == str(query_id)]

			# Calculate nDCG for the current query
			nDCG = self.queryNDCG(query_doc_ids_ordered, query_id, true_doc_ids, k)

			# Add nDCG to the count
			count += nDCG

		# Calculate mean nDCG
		meanNDCG = count / len(query_ids)

		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		count = 0  # Count of relevant documents retrieved
		p_sum = 0  # Sum of precisions at different retrieval positions

		for i in range(k):
			if query_doc_IDs_ordered[i] in 	true_doc_IDs:  # Check if retrieved document ID exists in true doc IDs
				count += 1

				# Calculate precision at position (i + 1) using your defined queryPrecision function
				precision_at_i = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, i + 1)
				p_sum += precision_at_i

		# Avoid division by zero
		avgPrecision = p_sum / count if count != 0 else 0
		
		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		count = 0

		for i in range(len(query_ids)):

			count += self.queryAveragePrecision(doc_IDs_ordered[i], \
									   query_ids[i],self.true_doc_id_list(query_ids[i],q_rels), k)
			
		meanAveragePrecision = count / len(query_ids)

		return meanAveragePrecision

