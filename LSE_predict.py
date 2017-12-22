import tensorflow as tf

class LSE_predict(object):

    def __init__(self, Wv, W, b):
        self.Wv = Wv
        self.W = W
        self.b = b
        
    def entityEmbedding(self, D):
        D = self.vocabulary.lookup(D)  # b x d x n x w
        t = tf.cast(tf.equal(D, tf.constant(0, dtype=tf.int64)), tf.float32)
        D = tf.cast(tf.cast(D, tf.float32) - t, tf.int64)
        D = tf.one_hot(D, self.vocab_size)  # b x d x n x w x V
        s = tf.reduce_sum(D, axis=4)  # b x d x n x w
        s = tf.reduce_sum(s, axis=3, keep_dims=True) + self.e  # b x d x n


        D = tf.reduce_sum(D, axis=3)  # b x d x n x V

        D = tf.divide(D, s)  # b x d x n x V
        D = tf.reduce_mean(D, axis=2)  # b x d x V
        D = tf.matmul(D, self.Wv)  # b x d x w_emb
        D = tf.tanh(tf.matmul(D, self.W) + self.b)  # b x d x e_emb

        return tf.reduce_sum(D, axis=1)  # b x e_emb

    # projects ngrams into entity embedding space
    # input is a b x w tensor
    def project(self, ngrams):
        D = self.vocabulary.lookup(ngrams)  # b x w
        t = tf.cast(tf.equal(D, tf.constant(0, dtype=tf.int64)), tf.float32)
        D = tf.cast(tf.cast(D, tf.float32) - t, tf.int64)
        D = tf.one_hot(D, self.vocab_size)  # b x w x V
        s = tf.reduce_sum(D, axis=2)  # b x w
        s = tf.reduce_sum(s, axis=1, keep_dims=True)  # b

        D = tf.reduce_sum(D, axis=1)  # b x V
        D = tf.divide(D, s)
        D = tf.expand_dims(D, 1)  # b x 1 x V
        D = tf.matmul(D, self.Wv)  # b x 1 x w_emb
        D = tf.tanh(tf.matmul(D, self.W) + self.b)  # b x 1 x e_emb
        return D