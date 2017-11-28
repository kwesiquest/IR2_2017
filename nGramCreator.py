import pandas as pd

def create_nGrams(n, text):
	nGrams = []
	words = text.split()
	i = 0
	while i+n <= len(words):
		nGrams.append(words[i:i+n])
		i += 1
	return nGrams

def create_ID_nGrams(n, pickle_file):
	productList = pd.read_pickle(pickle_file)
	id_nGrams = []
	size = 0
	for prod_id, reviewText in zip(productList["prod_id"], productList["reviewText"]):
		id_nGrams.append([prod_id, create_nGrams(n, reviewText)])
		size += 1
		if size >=128:
			break
	return pd.DataFrame(id_nGrams, columns=["prod_id", "reviewText"])

n = 3
pickle_file = "reviews_Home_and_Kitchen_5.pickle"
new_file = "reviews_Home_and_Kitchen_" + str(n) + "grams_small.pickle"

df = create_ID_nGrams(n, pickle_file)

df.to_pickle(new_file)