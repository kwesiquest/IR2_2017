import numpy as np

class NDCG_at_k:
    
    def __init__(self, k=10):
        max_relevance = [1] + [0] * k
        self.normalizing_factor = self.NDCG(max_relevance, k, norm=False) 
        
    def NDCG(self, ranking, k, norm):    
        discounted_gain = []
    
        for rank, relevance in enumerate(ranking[0:k]):
            discounted_gain.append(((2.**relevance)-1)/float(np.log2(rank + 2)))
        
        DG = sum(discounted_gain)
        if norm:
            return DG/self.normalizing_factor
        else: return DG

class precision_at_k:
    def __init__(self, k=10):
        self.k = k
    
    def precision(self, ranking):
        return float(np.sum(ranking))/self.k

