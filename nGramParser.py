### use this file by first creating an object "EntityDict(<str: pathToReview.json.gz>, <int: nGram size>)"
###
### then call the function "next_entity(<int: number of dissimilar entities>)" in a loop, which yields an 
### "returnPackage()" object, consisting of the entity's id (".entity_id"), its documents in a list (".docs")
### and a list of documents belonging to z different entities (".dissimilars") 

import gzip
import numpy as np
import argparse
from nGramTokenMap import tokenise

class returnPackage:
	#ngrams = nGrams * words
	#docs = docs * nGrams * words
	#dissimilars = entities * docs * nGrams * words
	def __init__(self):
		self.entity_id = ""
		self.ngrams = []
		self.docs = []
		self.dissimilars = []

class bigReturnPackage:
	#docs = docs * nGrams * words
	#similars = entities * docs * nGrams * words
	#dissimilars = entities * docs * nGrams * words
	def __init__(self):
		self.entity_ids = []
		self.docs = []
		self.similars = []
		self.dissimilars = []

class EntityDict:
	# creates a dictionary with entities as keys and all the corresponding documents (in  a list) as values
	# a documents is a list of ngrams
	# ngrams are a list of strings
	def __init__(self, path, n, limited=False):
		self.entity_pointer = 0
		self.doc_pointer = 0
		self.path = path
		self.n = n
		self.entity_dict = {}
		self.continuous = True
		print("creating new dictionary...")
		for doc in self.parse():
			nGrams = create_nGrams(doc['reviewText'], self.n)
			entity_id = doc['asin']
			try:
				self.entity_dict[entity_id].append(nGrams)
			except:
				if limited and (len(self.entity_dict.keys()) >= 4):
					break
				self.entity_dict[entity_id] = [nGrams]
		self.entities = list(self.entity_dict.keys())
		print("created entity dictionary containing", len(self.entities), "entities.")

	#based on code from: http://jmcauley.ucsd.edu/data/amazon/links.html
	def parse(self):
		g = gzip.open(self.path, 'rb')
		for l in g:
			yield eval(l)



	def get_random_id(self, entity_ids=["0"]):
		g = gzip.open(self.path, 'rb')
		different_id = np.random.choice(self.entities)
		while different_id in entity_ids:
			different_id = np.random.choice(self.entities)
		return different_id

	def get_random_doc(self, entity_id):
		index = np.random.randint(len(self.entity_dict[entity_id]))
		return(self.entity_dict[entity_id][index])

	def get_dissimilars(self, entity_id, z):
		dissimilars = []
		entity_ids = [entity_id]
		for _ in range(0, z):
			dissimilar_id = self.get_random_id(entity_ids)
			entity_ids.append(dissimilar_id)
			# dissimilars += self.entity_dict[self.get_random_id(entity_id)] #dissimilars is a list of documents
			dissimilars.append(self.entity_dict[dissimilar_id]) #dissimilars is a list of entities
		return dissimilars

	def move_pointers(self):
		if self.doc_pointer == len(self.entity_dict[self.entities[self.entity_pointer]])-1:
			self.doc_pointer = 0
			self.entity_pointer += 1
			if(self.entity_pointer >= len(self.entities)):
				self.entity_pointer = 0
		else:
			self.doc_pointer += 1

	# z = number of non-similar entities
	# nDocs = number of return values
	def get_random_batch(self, z, nDocs):
		return_value = bigReturnPackage()
		for n in range(0, nDocs):
			return_value.entity_ids.append(self.get_random_id())
			# return_value.docs.append(self.get_random_doc(return_value.entity_ids[n])[:])
			doc = self.get_random_doc(return_value.entity_ids[n])[:]
			return_value.docs.append(doc[np.random.randint(len(doc))])
			return_value.similars.append(self.entity_dict[return_value.entity_ids[n]][:])
			return_value.dissimilars.append(self.get_dissimilars(return_value.entity_ids[n], z)[:])
		return return_value

	# z = number of non-similar entities
	def next_doc_continuous(self, z):
		return_value = returnPackage()
		return_value.entity_id = self.entities[self.entity_pointer]
		return_value.ngrams = self.entity_dict[return_value.entity_id][self.doc_pointer]
		return_value.docs = self.entity_dict[return_value.entity_id]
		return_value.dissimilars = self.get_dissimilars(return_value.entity_id, z)
		self.move_pointers()
		return return_value

	# z = number of non-similar entities
	def next_entity(self, z):
		for entity_id in self.entities:
			return_value = returnPackage()
			return_value.entity_id = entity_id
			return_value.docs = self.entity_dict[entity_id]
			return_value.dissimilars = self.get_dissimilars(entity_id, z)
			yield return_value

def create_nGrams(text, n):
	nGrams = []
	words = tokenise(text)
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

	parser = EntityDict(path, n, True)

	values = parser.get_random_batch(4, 15)

	for value, docs in zip(values.entity_ids, values.dissimilars):
		print(value, len(docs))


	# i = 0
	# for entity in parser.next_entity(2):
	# for j in range(0, 10):
	# 	i+= 1
	# 	plur = parser.next_doc_continuous(2)
	# 	print(i, plur.entity_id)

