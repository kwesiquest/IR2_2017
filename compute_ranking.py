import tensorflow as tf
import evaluation as eval
import numpy as np
import nGramParser as ngp

class compute_ranking:
    def __init__(self):
        self.ndcg = eval.NDCG_at_k()


def get_relevance(entities, query_embed):
    ranking = []
    for e in entities:
        # (relevance label, cosine_similarity)
        ranking.append((tf.sigmoid(tf.tensordot(e, query_embed)), tf.losses.cosine_distance(e, query_embed)))

    # sort ranking by cosine similarity
    return np.array(sorted(ranking, key=lambda x: x[1]))



def compute(queries, entities):

    scores = []

    for query in queries:

        ranking = get_relevance(entities, query)

        scores.append(self.ndcg.NDCG(ranking[:, 0], norm=True))

    print(np.mean(scores))



