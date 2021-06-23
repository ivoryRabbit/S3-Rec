import fire
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from collections import defaultdict
from models import S3Rec
from utils import load_user_item
from generators import test_generator


class evaluation:
    def run(self, user_item_fname, use_pretrained):

        user_item = load_user_item(user_item_fname)

        n_user = user_item.user_id.nunique()
        n_item = user_item.item_list.explode().nunique()
        max_item_len = 50
        latent_dim = 64

        batch_size = 256
        steps_per_epoch = n_user // batch_size + 1
        test_gen = test_generator(user_item, n_user, n_item, max_item_len, batch_size)

        model = S3Rec(
            n_user, n_item, max_item_len, latent_dim, n_layer=2, n_head=2, n_ffn_unit=latent_dim
        )
        if use_pretrained:
            model.load_weights("weights/finetuned")
        else:
            model.load_weights("weights/non_pretrained")
        # model.summary()

        metrics = defaultdict(float)
        ndcg_weights = np.reciprocal(np.log2(np.arange(2, 102)))  # ndcg weights
        mrr_weights = 1.0 / np.arange(1, 101)  # mrr weights

        print("start evaluation...")
        for query, cand in tqdm(test_gen, total=steps_per_epoch):
            output = model(query)[:, -1, :]
            cand_embed = model.item_embedding_layer(cand)
            score = tf.einsum("nh,nlh->nl", output, cand_embed).numpy()
            rec = np.argsort(score, axis=1)[:, ::-1]  # recommended items by model
            rel = np.float32(rec == 0)
            for k in [1, 5, 10]:
                rel_k = rel[:, :k]

                HR = np.sum(rel_k) / n_user
                NDCG = (
                    np.sum(rel_k * ndcg_weights[:k]) / n_user
                )  # since # of relevants is one, there is no reason to find IDCG
                metrics[f"HR@{k}"] += HR
                if k > 1:
                    metrics[f"NDCG@{k}"] += NDCG
            metrics["MRR"] += np.sum(rel * mrr_weights) / n_user
        print("evaluation finished...")

        print("save result...")
        with open("eval_result.pkl", "wb") as f:
            pickle.dump(metrics, f)
        f.close()
        print("finished...")


if __name__ == "__main__":
    fire.Fire(evaluation)
