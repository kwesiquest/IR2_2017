import tensorflow as tf
import evaluation as eval
import numpy as np
import nGramParser as ngp


from nGramParser import EntityDict as edict
from nGramTokenMap2 import Vocab
from LSE import LSE as LSE
import operator


def get_relevance(entities, query_embed):
    ranking = []
    for e in entities:
        # (relevance label, cosine_similarity)
        ranking.append((tf.sigmoid(tf.tensordot(e, query_embed)), tf.losses.cosine_distance(e, query_embed)))

    # sort ranking by cosine similarity
    return np.array(sorted(ranking, key=lambda x: x[1]))


def project_entities(path, entities):

    projected = dict.fromkeys(entities,[])

    with tf.Session() as sess:

        new_saver = tf.train.import_meta_graph('../saves/model_basic.ckpt.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('../saves/'))
        graph = tf.get_default_graph()

        Wv = graph.get_operation_by_name('Wv')
        W = graph.get_operation_by_name('W')
        b = graph.get_operation_by_name('b')

        print("Model restored.")

        #for key, new_entity in entities.iteritems():
        #    projected[key] = sess.run(output, feed_dict={input:new_entity})


if __name__ == '__main__':

    ndcg = eval.NDCG_at_k()
    entities = edict('../reviews_Home_and_Kitchen_5.json.gz', 2, True).entity_dict
    project_entities('../saves/model_basic.ckpt', entities)

    '''
    scores = []
    queries = ['a', 'b'] # this needs to be a pickle of a list of queries
    entities = ["entity1", "entity2"] # this needs to be a pickle of a list of entities

    for query in queries:
        query_embed = lse.entityembedding(query)

        ranking = get_relevance(entities, query)

        scores.append(ndcg.NDCG(ranking[:, 0], norm=True))

    print(np.mean(scores))
    '''


