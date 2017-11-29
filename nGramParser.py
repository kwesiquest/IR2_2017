#based on code from: http://jmcauley.ucsd.edu/data/amazon/links.html

import pandas as pd
import gzip
import numpy as np
import argparse

from os.path import isfile

class doc_parser:
	def __init__(self, path, n):
		self.path = path
		self.n = n
		self.entity_docs = {}
		if isfile(path+".csv"):
			print("found existing .csv")
			pd.read_csv(path+".csv")
		else:
			print("creating new dictionary")
			for doc in self.parse():
				nGrams = create_nGrams(doc['reviewText'], self.n)
				entity_id = doc['asin']
				try:
					self.entity_docs[entity_id].append(nGrams)
				except:
					self.entity_docs[entity_id] = [nGrams]
			print("created entity dictionary.")
			entity_data = pd.DataFrame(columns=["prod_id", "docs"])
			entity_data["prod_id"] = self.entity_docs.keys()
			entity_data["docs"] = self.entity_docs.values()
			entity_data.to_csv(path + ".csv")

		# self.n_entities = len(self.entity_docs.keys())

	def parse(self):
		g = gzip.open(self.path, 'rb')
		for l in g:
			yield eval(l)

	def get_random_id(self, entity_id):
		g = gzip.open(self.path, 'rb')
		different_id = entity_id
		while(entity_id == different_id):
			different_id = np.random.choice(self.entity_docs.keys())
		return different_id

	# z = number of non-similar entities
	# def parse_batch(self, z):
		# for entity_id in self.



	# def parse_nGrams(self, n):
	# 	g = gzip.open(self.path, 'rb')
	# 	for l in g:
	# 		data = eval(l)
	# 		for nGram in create_nGrams(data['reviewText'], n):
	# 			yield nGram

	# def parse_batch(self, n):
	# 	g = gzip.open(self.path, 'rb')
	# 	for l in g:
	# 		data = eval(l)
	# 		nGrams = create_nGrams(data['reviewText'], n)
	# 		yield {'entity_id':data['asin'], 'nGrams':nGrams, 'dissimilars':}

def create_nGrams(text, n):
	nGrams = []
	words = text.split()
	i = 0
	while i+n <= len(words):
		nGrams.append(words[i:i+n])
		i += 1
	return nGrams

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('-path', type=str, help='path to data', default="reviews_Home_and_Kitchen_5.json.gz" )
	parser.add_argument('-n', type=int, help='specify n for ngrams', default=4)

	 
	FLAGS, unparsed = parser.parse_known_args()
	path = FLAGS.path
	n = FLAGS.n
     
	# i = 0
	# for nGram in parse_nGrams(path, n):
	# 	print(nGram)
	# 	if i >= 500:
	# 		break
	# 	else:
	# 		i+= 1

	parser = doc_parser(path, n)



	# print(i, str(n) + "-grams processed.")
