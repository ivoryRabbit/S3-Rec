import fire
import tensorflow as tf
from models import S3Rec, AAP_model, MIP_model, MAP_model, SP_model
from utils import load_user_item, load_item_attr
from generators import (
    AAP_generator, MIP_generator, MAP_generator, SP_generator
)

class pretrain:
    def run(self, user_item_fname, item_attr_fname, verbose):
        user_item = load_user_item(user_item_fname)
        item_attr = load_item_attr(item_attr_fname)

        n_user = user_item.user_id.nunique()
        n_item = item_attr.item_id.nunique()
        n_attr = item_attr.attr_list.explode().nunique()
        max_item_len = 50
        max_attr_len = item_attr.attr_list.apply(len).max()
        latent_dim = 64

        batch_size = 200
        user_per_batch = n_user // batch_size + 1
        item_per_batch = n_item // batch_size + 1

#        optimizer = tf.keras.optimizers.Adam(1e-3) # <- does this make models share adam's moment weights?
        attr_embedding_layer = tf.keras.layers.Embedding(
            n_attr+1, latent_dim, mask_zero = True, 
            input_length =  max_attr_len, name = 'attr_embedding'
        )

        base_model = S3Rec(
            n_user, n_item, max_item_len, latent_dim,
            n_layer = 2, n_head = 2, n_ffn_unit = latent_dim,
            dropout_rate = 0.1, epsilon = 1e-3
        )

        # data generators
        aap_train_gen = AAP_generator(item_attr, n_item, n_attr, batch_size)
        mip_train_gen = MIP_generator(user_item, n_user, n_item, max_item_len, batch_size)
        map_train_gen = MAP_generator(user_item, item_attr, n_user, n_attr, max_item_len, batch_size)
        sp_train_gen = SP_generator(user_item, n_user, n_item, max_item_len, batch_size)

        # set models
        aap_model = AAP_model(n_attr, base_model, attr_embedding_layer, loss_weight = 0.2)
        aap_model.compile(optimizer = tf.keras.optimizers.Adam(1e-3))

        mip_model = MIP_model(base_model, loss_weight = 1.0)
        mip_model.compile(optimizer = tf.keras.optimizers.Adam(1e-3))

        map_model = MAP_model(n_attr, base_model, attr_embedding_layer, loss_weight = 1.0)
        map_model.compile(optimizer = tf.keras.optimizers.Adam(1e-3))

        sp_model = SP_model(base_model, loss_weight = 0.5)
        sp_model.compile(optimizer = tf.keras.optimizers.Adam(1e-3))

        # total epochs = 100
        epochs = 1
        for s in range(100):
            print(f'{s+1}-step: start pretraining...')
            # Associated Attribute Prediction
            aap_model.fit(aap_train_gen, epochs = epochs, steps_per_epoch = item_per_batch, verbose = verbose)

            # Masked Attribute Prediction
            mip_model.fit(mip_train_gen, epochs = epochs, steps_per_epoch = user_per_batch, verbose = verbose)

            # Masked Attribute Prediction
            map_model.fit(map_train_gen, epochs = epochs, steps_per_epoch = user_per_batch, verbose = verbose)

            # Segment Predict
            sp_model.fit(sp_train_gen, epochs = epochs, steps_per_epoch = user_per_batch, verbose = verbose)
        base_model.save_weights('weights/pretrained')
        print('finished...')

if __name__ == '__main__':
    fire.Fire(pretrain)