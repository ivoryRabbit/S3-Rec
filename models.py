import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Embedding, Dense
from layers import Encoder, PositionEncoder

class S3Rec(tf.keras.Model):
    def __init__(self,
                 n_user,
                 n_item,
                 max_item_len,
                 latent_dim, 
                 n_layer, 
                 n_head, 
                 n_ffn_unit, # the number of units for feed forward network 
                 dropout_rate = 0.0,
                 epsilon = 1e-3):
        
        super(S3Rec, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.max_item_len = max_item_len
        self.latent_dim = latent_dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_ffn_unit = n_ffn_unit
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon

        self.item_embedding_layer = Embedding(n_item+1, 
                                              latent_dim, 
                                              mask_zero = True, # mask value = 0
                                              input_length = max_item_len,
                                              name = 'item_embedding')
        self.position_encoding_layer = PositionEncoder()
        self.item_encoding_layer = Encoder(n_layer, 
                                           n_head, 
                                           n_ffn_unit, 
                                           dropout_rate, 
                                           epsilon)

        # following two lines are just for activating "summary" attribute
        _ = self.call(Input(shape = (self.max_item_len, )), training = True)
        self.build(input_shape = (None, self.max_item_len))

    def call(self, inputs, mask = None, training = False):
        item_embed = self.item_embedding_layer(inputs)
        embed = self.position_encoding_layer(item_embed)
        if mask is None:
            mask = self.get_attn_mask(inputs)
        return self.item_encoding_layer(embed, mask, training = training)

    def get_score(self, outputs, targ):
        '''
        get similarities between the representatives of sequences and the embeddings of target items
        '''
        embed = self.item_embedding_layer(targ) # (batch_size, max_item_len, latent_dim)
        return tf.reduce_sum(tf.multiply(outputs, embed), axis = -1) # inner product w.r.t max_item_len (batch_size, max_item_len)

    def train_step(self, data):
        item, pos_targ, neg_targ = data
        look_mask = self.get_look_mask(item) # mask all next items
        loss_mask = self.get_mask(item) # ignore zero paddings
        with tf.GradientTape() as tape:
            outputs = self(item, look_mask, training = True)
            pos_score = self.get_score(outputs, pos_targ)
            neg_score = self.get_score(outputs, neg_targ)
            loss = tf.reduce_sum(self.loss(pos_score, neg_score, loss_mask), axis = 0)
            
        grad = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return {'loss': loss}

    def test_step(self, data):
        query, cand = data
        batch_size = tf.shape(query)[0]
        output = self(query)[:, -1, :]
        cand_embed = self.item_embedding_layer(cand)
        score = tf.einsum('nh,nlh->nl', output, cand_embed)

        k = 10
        top_k = tf.argsort(score, axis = 1, direction = 'DESCENDING')[:, :k]
        rel = tf.cast(top_k == 0, tf.float32) # relevant item is located at "0". remark valid_generator function
        batch_size = tf.cast(batch_size, tf.float32)

        weight = np.reciprocal(np.log2(np.arange(2, k+2)))
        weight = tf.constant(weight, dtype = tf.float32)

        HR = tf.reduce_sum(rel) / batch_size
        NDCG = tf.reduce_sum(rel * weight) / batch_size
        return {f'HR@{k}': HR, f'NDCG@{k}': NDCG}

    def get_mask(self, inputs): # calculating loss, ignore zero paddings
        return tf.cast(tf.math.greater(inputs, 0), dtype = tf.float32)

    def get_attn_mask(self, inputs): # mask zero paddings for MultiHeadAttention
        attn_mask = 1.0 - self.get_mask(inputs)
        return attn_mask[:, None, None, :] # (batch_size, 1, 1, max_item_len)

    def get_look_mask(self, inputs): # mask all next items for MultiHeadAttention
        attn_mask = 1.0 - self.get_attn_mask(inputs)
        ltri_mask = tf.ones((1, 1, self.max_item_len, self.max_item_len), dtype = tf.float32)
        ltri_mask = tf.linalg.band_part(ltri_mask, -1, 0) # lower triangular matrix
        return 1.0 - (attn_mask * ltri_mask) # (batch_size, 1, max_item_len, seq_len)

    def get_item_embedding(self):
        total_idx = tf.range(1, self.n_item+1)
        return self.item_embedding_layer(total_idx)

class AAP_model(tf.keras.Model):
    def __init__(self, n_attr, base_model, attr_embedding_layer, loss_weight):
        super(AAP_model, self).__init__()
        self.n_attr = n_attr
        self.item_embedding_layer = base_model.item_embedding_layer
        self.attr_embedding_layer = attr_embedding_layer
        self.loss_weight = loss_weight
        self.W_aap = Dense(base_model.latent_dim, name = 'AAP_dense')

    def call(self, inputs):
        e_i = self.item_embedding_layer(inputs)
        return self.W_aap(e_i)

    def get_attr_embedding(self):
        total_idx = tf.range(1, self.n_attr+1)
        return self.attr_embedding_layer(total_idx)

    def get_score(self, outputs): 
        attr_embed = self.get_attr_embedding() # the complement set of A_i == the negative samples of item i
        return tf.matmul(outputs, attr_embed, transpose_b = True) # (batch_size, max_item_len, n_attr)

    def get_loss(self, pos, neg):
        bce = tf.keras.losses.BinaryCrossentropy(from_logits = True, reduction = tf.keras.losses.Reduction.NONE)
        return bce(pos, neg)

    def train_step(self, data):
        item, attr = data
        with tf.GradientTape() as tape:
            outputs = self(item)
            score = self.get_score(outputs)
            loss = self.loss_weight * tf.reduce_sum(self.get_loss(attr, score), axis = 0)
        grad = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return {'loss': loss}

class MIP_model(tf.keras.Model):
    def __init__(self, base_model, loss_weight):
        super(MIP_model, self).__init__()
        self.base_model = base_model
        self.loss_weight = loss_weight
        self.W_mip = Dense(base_model.latent_dim, name = 'MIP_dense')

    def call(self, inputs, training = False):
        f_t = self.base_model(inputs, training = training)
        return self.W_mip(f_t)

    def get_score(self, outputs, targ):
        e_i = self.base_model.item_embedding_layer(targ) # (batch_size, max_item_len, latent_dim)
        return tf.reduce_sum(tf.multiply(outputs, e_i), axis = -1) # inner product (batch_size, max_item_len)

    def get_loss(self, pos, neg, mask):
        bpr = K.log(tf.nn.sigmoid(pos - neg)) # (batch_size, max_item_len)
        # bce = K.log(tf.nn.sigmoid(pos)) + K.log(tf.nn.sigmoid(1-neg))
        return -tf.reduce_sum(bpr * mask, axis = 1) / tf.reduce_sum(mask, axis = 1) # (batch_size, )

    def train_step(self, data):
        masked, pos_targ, neg_targ = data
        loss_mask = self.base_model.get_mask(pos_targ)
        with tf.GradientTape() as tape:
            outputs = self(masked, training = True)
            pos_score = self.get_score(outputs, pos_targ)
            neg_score = self.get_score(outputs, neg_targ)

            loss = self.loss_weight * tf.reduce_sum(self.get_loss(pos_score, neg_score, loss_mask), axis = 0)
            
        grad = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return {'loss': loss}

class MAP_model(tf.keras.Model):
    def __init__(self, n_attr, base_model, attr_embedding_layer, loss_weight):
        super(MAP_model, self).__init__()
        self.n_attr = n_attr
        self.base_model = base_model
        self.attr_embedding_layer = attr_embedding_layer
        self.loss_weight = loss_weight
        self.W_map = Dense(base_model.latent_dim, name = 'MAP_dense')

    def call(self, inputs, training = False):
        f_t = self.base_model(inputs, training = training)
        return self.W_map(f_t) # (batch_size, max_item_len, n_attr)

    def get_attr_embedding(self):
        total_idx = tf.range(1, self.n_attr+1)
        return self.attr_embedding_layer(total_idx) # (n_attr, latent_dim)

    def get_score(self, outputs):
        attr_embed = self.get_attr_embedding()
        return tf.matmul(outputs, attr_embed, transpose_b = True) # (batch_size, max_item_len, n_attr)

    def get_loss(self, true, pred, mask):
        bce = tf.keras.losses.BinaryCrossentropy(from_logits = True, reduction = tf.keras.losses.Reduction.NONE)
        return tf.reduce_sum(bce(true, pred) * mask, axis = 1) / tf.reduce_sum(mask, axis = 1)

    def train_step(self, data):
        item, attr = data
        loss_mask = self.base_model.get_mask(item)
        with tf.GradientTape() as tape:
            outputs = self(item, training = True)
            score = self.get_score(outputs)
            loss = self.loss_weight * tf.reduce_sum(self.get_loss(attr, score, loss_mask), axis = 0)
        grad = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return {'loss': loss}

class SP_model(tf.keras.Model):
    def __init__(self, base_model, loss_weight):
        super(SP_model, self).__init__()
        self.base_model = base_model
        self.loss_weight = loss_weight
        self.W_sp = Dense(base_model.latent_dim, name = 'SP_dense')

    def call(self, inputs, training = False):
        s = self.base_model(inputs, training = training)[:, -1, :] # the last position in a sequence
        return self.W_sp(s) # (batch_size, latent_dim)

    def get_score(self, outputs, seg, training = False):
        s_til = self.base_model(seg, training = training)[:, -1, :] # the last position in a sequence
        return tf.reduce_sum(tf.multiply(outputs, s_til), axis = 1) # inner product (batch_size, )
        
    def get_loss(self, pos, neg):
        bpr = K.log(tf.nn.sigmoid(pos - neg))
        # bce = K.log(tf.nn.sigmoid(pos)) + K.log(tf.nn.sigmoid(1-neg))
        return -bpr # (batch_size, )

    def train_step(self, data):
        masked, pos_seg, neg_seg = data
        with tf.GradientTape() as tape:
            outputs = self(masked, training = True)
            pos_score = self.get_score(outputs, pos_seg, training = True)
            neg_score = self.get_score(outputs, neg_seg, training = True)

            loss = self.loss_weight * tf.reduce_sum(self.get_loss(pos_score, neg_score), axis = 0)
            
        grad = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return {'loss': loss}