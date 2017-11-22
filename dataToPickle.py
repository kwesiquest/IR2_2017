#based on code from: http://jmcauley.ucsd.edu/data/amazon/links.html

import pandas as pd
import gzip
import numpy as np

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  df = []
  for d in parse(path):
    df.append([d['asin'], d['reviewText']])
  return pd.DataFrame(df, columns=["prod_id", "reviewText"])

path = "reviews_Home_and_Kitchen_5.json.gz"
pick = "reviews_Home_and_Kitchen_5.pickle"

df = getDF(path)

df.to_pickle(pick)


##read:
# df = pd.read_pickle(pick)
# print(df.shape)