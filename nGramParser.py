#based on code from: http://jmcauley.ucsd.edu/data/amazon/links.html

import pandas as pd
import gzip
import numpy as np

def parse(path):
	g = gzip.open(path, 'rb')
	for l in g:
		yield eval(l)

def parse_nGrams(path, n):
	g = gzip.open(path, 'rb')
	for l in g:
		data = eval(l)
		for nGram in create_nGrams(data['reviewText'], n):
			yield nGram

def parse_batch(path, n):
	g = gzip.open(path, 'rb')
	for l in g:
		data = eval(l)
		nGrams = create_nGrams(data['reviewText'], n):
			yield {'entity_id':data['asin'], 'nGrams':nGrams}


def create_nGrams(text, n):
	nGrams = []
	words = text.split()
	i = 0
	while i+n <= len(words):
		nGrams.append(words[i:i+n])
		i += 1
	return nGrams

if __name__ == '__main__':
	path = "reviews_Home_and_Kitchen_5.json.gz"
	n = 4

	i = 0
	for nGram in parse_nGrams(path, n):
		print(nGram)
		if i >= 500:
			break
		else:
			i+= 1

	print(i, str(n) + "-grams processed.")