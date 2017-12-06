import LSE
import tensorflow as tf
import evaluation as eval
import numpy as np

def get_relevance(entities, query_embed):
    ranking = []
    for e in entities:
        # (relevance label, cosine_similarity)
        ranking.append((tf.sigmoid(tf.tensordot(e, query_embed)), tf.losses.cosine_distance(e, query_embed)))

    # sort ranking by cosine similarity
    return np.array(sorted(ranking, key=lambda x: x[1]))



if __name__ == '__main__':
    lse = LSE()
    ndcg = eval.NDCG_at_k()

    scores = []
    queries = ['a', 'b'] # this needs to be a pickle of a list of queries
    entities = ["entity1", "entity2"] # this needs to be a pickle of a list of entities

    for query in queries:
        query_embed = lse.entityembedding(query)

        ranking = get_relevance(entities, query)

        scores.append(ndcg.NDCG(ranking[:, 0], norm=True))

    print(np.mean(scores))


