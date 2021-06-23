import fire
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from models import S3Rec
from utils import load_user_item
from generators import train_generator, valid_generator


class finetune:
    def run(self, user_item_fname, use_pretrained, verbose):
        user_item = load_user_item(user_item_fname)

        n_user = user_item.user_id.nunique()
        n_item = user_item.item_list.explode().nunique()
        max_item_len = 50
        latent_dim = 64

        epochs = 40
        batch_size = 256
        steps_per_epoch = n_user // batch_size + 1

        train_gen = train_generator(user_item, n_user, n_item, max_item_len, batch_size)
        valid_gen = valid_generator(user_item, n_user, n_item, max_item_len, batch_size)

        def masked_loss(pos, neg, mask):
            # bce = K.log(tf.nn.sigmoid(pos)) + K.log(1 - tf.nn.sigmoid(neg)) # (batch_size, seq_len)
            bpr = K.log(tf.nn.sigmoid(pos - neg))
            return -tf.reduce_sum(bpr * mask, axis=-1) / tf.reduce_sum(
                mask, axis=-1
            )  # (batch_size, )

        optimizer = tf.keras.optimizers.Adam(1e-3)
        model = S3Rec(
            n_user,
            n_item,
            max_item_len,
            latent_dim,
            n_layer=2,
            n_head=2,
            n_ffn_unit=latent_dim,
            dropout_rate=0.1,
            epsilon=1e-3,
        )
        model.compile(optimizer=optimizer, loss=masked_loss)
        # model.summary()

        if use_pretrained:
            model.load_weights("weights/pretrained")
            checkpoint = "weights/finetuned"
        else:
            checkpoint = "weights/non_pretrained"

        model_checkpoint = ModelCheckpoint(
            checkpoint, monitor="val_NDCG@10", mode="max", save_best_only=False
        )

        print("start finetuning...")
        model.fit(
            train_gen,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=valid_gen,
            validation_steps=steps_per_epoch // 10,  # sampling 10% of dataset to validate
            callbacks=[model_checkpoint],
            verbose=verbose,
        )
        print("finetuning finished...")


if __name__ == "__main__":
    fire.Fire(finetune)
