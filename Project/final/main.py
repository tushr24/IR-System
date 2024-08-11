from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from informationRetrieval import InformationRetrieval
from evaluation import Evaluation
import time, os
from util import *


from sys import version_info
import argparse
import json
import matplotlib.pyplot as plt

# Input compatibility for Python 2 and Python 3
if version_info.major == 3:
    pass
elif version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass
else:
    print ("Unknown python version - input function not safe")


class SearchEngine:

	def __init__(self, args):
		self.args = args

		self.tokenizer = Tokenization()
		self.sentenceSegmenter = SentenceSegmentation()
		self.inflectionReducer = InflectionReduction()
		self.stopwordRemover = StopwordRemoval()

		self.informationRetriever = InformationRetrieval()
		self.evaluator = Evaluation()


	def segmentSentences(self, text):
		"""
		Call the required sentence segmenter
		"""
		if self.args.segmenter == "naive":
			return self.sentenceSegmenter.naive(text)
		elif self.args.segmenter == "punkt":
			return self.sentenceSegmenter.punkt(text)

	def tokenize(self, text):
		"""
		Call the required tokenizer
		"""
		if self.args.tokenizer == "naive":
			return self.tokenizer.naive(text)
		elif self.args.tokenizer == "ptb":
			return self.tokenizer.pennTreeBank(text)

	def reduceInflection(self, text):
		"""
		Call the required stemmer/lemmatizer
		"""
		return self.inflectionReducer.reduce(text)

	def removeStopwords(self, text):
		"""
		Call the required stopword remover
		"""
		return self.stopwordRemover.fromList(text)


	def preprocessQueries(self, queries):
		"""
		Preprocess the queries - segment, tokenize, stem/lemmatize and remove stopwords
		"""

		# Segment queries
		segmentedQueries = []
		for query in queries:
			segmentedQuery = self.segmentSentences(query)
			segmentedQueries.append(segmentedQuery)
		json.dump(segmentedQueries, open(self.args.out_folder + "segmented_queries.txt", 'w'))
		# Tokenize queries
		tokenizedQueries = []
		for query in segmentedQueries:
			tokenizedQuery = self.tokenize(query)
			tokenizedQueries.append(tokenizedQuery)
		json.dump(tokenizedQueries, open(self.args.out_folder + "tokenized_queries.txt", 'w'))
		# Stem/Lemmatize queries
		reducedQueries = []
		for query in tokenizedQueries:
			reducedQuery = self.reduceInflection(query)
			reducedQueries.append(reducedQuery)
		json.dump(reducedQueries, open(self.args.out_folder + "reduced_queries.txt", 'w'))
		# Remove stopwords from queries
		stopwordRemovedQueries = []
		for query in reducedQueries:
			stopwordRemovedQuery = self.removeStopwords(query)
			stopwordRemovedQueries.append(stopwordRemovedQuery)
		json.dump(stopwordRemovedQueries, open(self.args.out_folder + "stopword_removed_queries.txt", 'w'))

		preprocessedQueries = stopwordRemovedQueries
		return preprocessedQueries

	def preprocessDocs(self, docs):
		"""
		Preprocess the documents
		"""
		
		# Segment docs
		segmentedDocs = []
		for doc in docs:
			segmentedDoc = self.segmentSentences(doc)
			segmentedDocs.append(segmentedDoc)
		json.dump(segmentedDocs, open(self.args.out_folder + "segmented_docs.txt", 'w'))
		# Tokenize docs
		tokenizedDocs = []
		for doc in segmentedDocs:
			tokenizedDoc = self.tokenize(doc)
			tokenizedDocs.append(tokenizedDoc)
		json.dump(tokenizedDocs, open(self.args.out_folder + "tokenized_docs.txt", 'w'))
		# Stem/Lemmatize docs
		reducedDocs = []
		for doc in tokenizedDocs:
			reducedDoc = self.reduceInflection(doc)
			reducedDocs.append(reducedDoc)
		json.dump(reducedDocs, open(self.args.out_folder + "reduced_docs.txt", 'w'))
		# Remove stopwords from docs
		stopwordRemovedDocs = []
		for doc in reducedDocs:
			stopwordRemovedDoc = self.removeStopwords(doc)
			stopwordRemovedDocs.append(stopwordRemovedDoc)
		json.dump(stopwordRemovedDocs, open(self.args.out_folder + "stopword_removed_docs.txt", 'w'))

		preprocessedDocs = stopwordRemovedDocs
		return preprocessedDocs


	def evaluateDataset(self):
		"""
		- preprocesses the queries and documents, stores in output folder
		- invokes the IR system
		- evaluates precision, recall, fscore, nDCG and MAP 
		  for all queries in the Cranfield dataset
		- produces graphs of the evaluation metrics in the output folder
		"""

		# Read queries
		queries_json = json.load(open(args.dataset + "cran_queries_wrong.json", 'r'))[:]
		query_ids, queries = [item["query number"] for item in queries_json], \
								[item["query"] for item in queries_json]

		
		start = time.time()

		# Process queries 
		processedQueries = self.preprocessQueries(queries)

		# Read documents
		docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]

		# Comment the below code when not using weights
		doc_ids, docs = [item["id"] for item in docs_json], \
							[item["body"] + "." + item["author"] +  "."  \
							+ item["author"] + "." + item["title"] for item in docs_json]

		# Comment the below code when using weights
		# doc_ids, docs = [item["id"] for item in docs_json], \
		# 						[item["body"] for item in docs_json]
		
		# Process documents
		processedDocs = self.preprocessDocs(docs)

		# Build document index
		self.informationRetriever.buildIndex(processedDocs, doc_ids)
		# Rank the documents for each query
		doc_IDs_ordered = self.informationRetriever.rank(processedQueries)

		# Read relevance judements
		qrels = json.load(open(args.dataset + "cran_qrels.json", 'r'))[:]

		# Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
		precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
		for k in range(1, 11):
			precision = self.evaluator.meanPrecision(
				doc_IDs_ordered, query_ids, qrels, k)
			precisions.append(precision)
			recall = self.evaluator.meanRecall(
				doc_IDs_ordered, query_ids, qrels, k)
			recalls.append(recall)
			fscore = self.evaluator.meanFscore(
				doc_IDs_ordered, query_ids, qrels, k)
			fscores.append(fscore)
			# print("Precision, Recall and F-score @ " +  
			# 	str(k) + " : " + str(precision) + ", " + str(recall) + 
			# 	", " + str(fscore))
			print(f"\nPrecision, Recall and F-score @ {k} : {precision:.4f}, {recall:.4f}, {fscore:.4f}")
			MAP = self.evaluator.meanAveragePrecision(
				doc_IDs_ordered, query_ids, qrels, k)
			MAPs.append(MAP)
			nDCG = self.evaluator.meanNDCG(
				doc_IDs_ordered, query_ids, qrels, k)
			nDCGs.append(nDCG)
			# print("MAP, nDCG @ " +  
			# 	str(k) + " : " + str(MAP) + ", " + str(nDCG))
			print(f"MAP, nDCG @ {k} : {MAP:.4f}, {nDCG[0]:.4f}")

		print(f'\nMean nDCG: {np.mean(nDCGs):.4f}\nMean MAP: {np.mean(MAPs):.4f}') 

		end = time.time()
		print(f"\nTime : {end - start:.2f}")

		# Plot the metrics and save plot 
		plt.plot(range(1, 11), precisions, 'b-', label="Precision")
		plt.plot(range(1, 11), recalls, 'g-', label="Recall")
		plt.plot(range(1, 11), fscores, 'r-', label="F-Score")
		plt.plot(range(1, 11), MAPs, 'm-', label="MAP")
		plt.plot(range(1, 11), nDCGs, 'k-', label="nDCG")
		plt.title("Evaluation Metrics - Cranfield Dataset")
		plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
		plt.xlabel("k")
		plt.ylabel("Metric value")
		plt.savefig(args.out_folder + "Evaluation_metrics.png", bbox_inches='tight')

		
	def handleCustomQuery(self):
		"""
		Take a custom query as input and return top five relevant documents
		"""

		#Get query
		print("> Enter query below:")
		query = input('> ')


		start = time.time()
		# Process documents
		processedQuery = self.preprocessQueries([query])[0]

		# Read documents
		docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
		doc_ids, docs= [item["id"] for item in docs_json], \
							[item["body"] + "." + item["author"] + "." + item["author"] + "." + item["title"] for item in docs_json]
		
		doc_names = [item["title"] for item in docs_json]
		# Process documents
		processedDocs = self.preprocessDocs(docs)

		# Build document index
		self.informationRetriever.buildIndex(processedDocs, doc_ids)
		# Rank the documents for the query
		doc_IDs_ordered = self.informationRetriever.rank([processedQuery])[0]

		# Print the IDs of first five documents
		print("\nTop five documents : ")
		for id_ in doc_IDs_ordered[:5]:
			print(f'\n> Doc ID: {id_}')
			print(f'> Doc title: {doc_names[id_ - 1]}')
		end = time.time()
		print(f"\nTime: {end - start:.2f}")


	def just_evaluate(self, K=400, w =1.0):
		"""
		- preprocesses the queries and documents, stores in output folder
		- invokes the IR system
		- evaluates precision, recall, fscore, nDCG and MAP 
		  for all queries in the Cranfield dataset
		- produces graphs of the evaluation metrics in the output folder
		"""

		# Read queries
		queries_json = json.load(open(args.dataset + "cran_queries.json", 'r'))[:]
		query_ids, queries = [item["query number"] for item in queries_json], \
								[item["query"] for item in queries_json]
		# Process queries 
		processedQueries = self.preprocessQueries(queries)

		# Read documents
		docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
		doc_ids, docs = [item["id"] for item in docs_json], \
							[item["body"] + "." +item["author"] +"." \
							+ item["author"] + "."+ item["title"] for item in docs_json]
		# Process documents,item["author"]
		processedDocs = self.preprocessDocs(docs)

		# Build document index
		self.informationRetriever.buildIndex(processedDocs, doc_ids)
		# Rank the documents for each query
		doc_IDs_ordered = self.informationRetriever.rank(processedQueries, K, w)

		# Read relevance judements
		qrels = json.load(open(args.dataset + "cran_qrels.json", 'r'))[:]

		# Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
		precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
		for k in range(1, 11):
			precision = self.evaluator.meanPrecision(
				doc_IDs_ordered, query_ids, qrels, k)
			precisions.append(precision)
			recall = self.evaluator.meanRecall(
				doc_IDs_ordered, query_ids, qrels, k)
			recalls.append(recall)
			fscore = self.evaluator.meanFscore(
				doc_IDs_ordered, query_ids, qrels, k)
			fscores.append(fscore)
			MAP = self.evaluator.meanAveragePrecision(
				doc_IDs_ordered, query_ids, qrels, k)
			MAPs.append(MAP)
			nDCG = self.evaluator.meanNDCG(
				doc_IDs_ordered, query_ids, qrels, k)
			nDCGs.append(nDCG)

		return(np.max(nDCGs), np.max(fscores))
			


	def gridsearch(self, plot_search = False):
		"""
		Grid search on max nDCG@k and F-score
		"""
		
		w_dict_k_scores = {}
		max_k = 1000
		min_k = 100
		step = 100
		ws = [1.0, 1.5, 2.0]
		count = 0

		for w in ws:
			nDCGs = []
			fscores = []
			ks = []
			for i,k in enumerate(np.arange(min_k,max_k,step)):
				count += 1
				print(f"Evaluating {count} of {((max_k-min_k)//step)*len(ws)}", end="\r")

				nDCG, fscore = self.just_evaluate(k, w)

				ks.append(k)
				nDCGs.append(nDCG)
				fscores.append(fscore)
				print(f'( k = {min_k+ step*i} , w = {w} ) : nDCG = {nDCG:.4f}')
			w_dict_k_scores[w] = [ks, nDCGs, fscores]
            
		if plot_search:
			os.makedirs(args.out_folder + f"gridsearch/", exist_ok=True)

			# Plot for nDCGs
			for w, (ks, nDCGs, _) in w_dict_k_scores.items():
				plt.plot(ks, nDCGs, '^-', label=f'w = {w}')  # Label with weight value

			plt.xlabel('Number of components retained')
			plt.ylabel('nDCG')
			plt.title('nDCG for Different Weights')
			plt.legend()
			plt.grid(True)  # Add grid for better readability
			plt.savefig(args.out_folder + "gridsearch/grid_search_nDCG.png")
			plt.clf()  # Clear the plot for the next figure

			# Plot for F-scores
			for w, (_, _, fscores) in w_dict_k_scores.items():
				plt.plot(ks, fscores, '^-', label=f'w={w}')  # Label with weight value

			plt.xlabel('Number of components retained')
			plt.ylabel('F-score')
			plt.title('F-score for Different Weights')
			plt.legend()
			plt.grid(True)  # Add grid for better readability
			plt.savefig(args.out_folder + "gridsearch/grid_search_Fscore.png")


		return ks, nDCGs, fscores



if __name__ == "__main__":

	# Create an argument parser
	parser = argparse.ArgumentParser(description='main.py')

	# Tunable parameters as external arguments
	parser.add_argument('-dataset', default = "cranfield/", 
						help = "Path to the dataset folder")
	parser.add_argument('-out_folder', default = "output/", 
						help = "Path to output folder")
	parser.add_argument('-segmenter', default = "naive",
	                    help = "Sentence Segmenter Type [naive|punkt]")
	parser.add_argument('-tokenizer',  default = "naive",
	                    help = "Tokenizer Type [naive|ptb]")
	parser.add_argument('-custom', action = "store_true", 
						help = "Take custom query as input")
	parser.add_argument('-grid', action = "store_true", 
						help = "To do GridSearch")
	
	# Parse the input arguments
	args = parser.parse_args()

	# Create an instance of the Search Engine
	searchEngine = SearchEngine(args)

	#Either handle query from user or evaluate on the complete dataset 
	if args.custom:
		searchEngine.handleCustomQuery()
	elif args.grid:
		searchEngine.gridsearch(plot_search = True)
	else:
		searchEngine.evaluateDataset()
