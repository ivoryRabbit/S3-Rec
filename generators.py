import numpy as np
import pandas as pd
from typing import Tuple
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import neg_sampling, shuffle, batch_slice, one_hot_encoding, make_mask, make_segment


def train_generator(
    user_item: pd.DataFrame, n_user: int, n_item: int, max_item_len: int, batch_size: int
) -> Tuple[np.ndarray]:
    """
    for each user, generate the indices of items interacted with the user
    and its positive or negative target items that pre-padded by zeros

    a batch of items: [0, 0, 1, 2, 3]
    a batch of positive targets: [0, 0, 2, 3, 4]
    a batch of negative targets: [0, 0, 97, 98, 99]
    """
    pad_seq = lambda d: pad_sequences(d, max_item_len, padding="pre")
    neg_sam = lambda x: neg_sampling(x, n_item, 1)

    train = user_item.item_list.apply(lambda x: x[:-2])
    item = train.apply(lambda x: x[:-1]).to_numpy()
    pos_targ = train.apply(lambda x: x[1:]).to_numpy()

    while True:
        neg_targ = train.apply(neg_sam).to_numpy()  # initialize negative sampling for each epoch
        item, pos_targ, neg_targ = shuffle(n_user, item, pos_targ, neg_targ)
        batch_step = int(np.ceil(n_user / batch_size))
        for step in range(batch_step):
            lower = batch_size * step
            upper = batch_size + lower

            i_b, p_b, n_b = batch_slice(lower, upper, item, pos_targ, neg_targ)
            yield tuple(map(pad_seq, (i_b, p_b, n_b)))


def valid_generator(
    user_item: pd.DataFrame, n_user: int, n_item: int, max_item_len: int, batch_size: int
) -> Tuple[np.ndarray]:
    """
    for each user, generate indices of items interacted with the user
    and its index of target item with 99 of negative samples

    a batch of query sequence of items: [0, 1, 2, 3, 4]
    a batch of 100 candidates: [5, 101, ..., 199]
    """
    pad_seq = lambda d: pad_sequences(d, max_item_len, padding="pre")
    neg_sam = lambda d: neg_sampling(d, n_item, ns=99)

    valid = user_item.item_list.apply(lambda x: x[:-1])
    valid_q = valid.apply(lambda x: x[:-1]).to_numpy()  # leave-one-out
    valid_c = valid.apply(lambda x: [x[-1]]) + valid.apply(neg_sam)  # add 99 of negative samples
    valid_c = np.vstack(valid_c)

    while True:
        valid_q, valid_c = shuffle(n_user, valid_q, valid_c)
        batch_step = int(np.ceil(n_user / batch_size))
        for step in range(batch_step):
            lower = batch_size * step
            upper = batch_size + lower

            q, c = batch_slice(lower, upper, valid_q, valid_c)
            yield pad_seq(q), c


def test_generator(
    user_item: pd.DataFrame, n_user: int, n_item: int, max_item_len: int, batch_size: int
) -> Tuple[np.ndarray]:
    """
    for each user, generate indices of items interacted with the user
    and its index of target item with 99 of negative samples

    a batch of query sequences of items: [1, 2, 3, 4, 5]
    a batch of 100 candidates: [6, 101, ..., 199]
    """
    pad_seq = lambda d: pad_sequences(d, max_item_len, padding="pre")
    neg_sam = lambda d: neg_sampling(d, n_item, ns=99)

    test = user_item.item_list
    test_q = test.apply(lambda x: x[:-1]).to_numpy()  # leave-one-out
    test_c = test.apply(lambda x: [x[-1]]) + test.apply(neg_sam)  # add 99 of negative samples
    test_c = np.vstack(test_c)

    batch_step = int(np.ceil(n_user / batch_size))
    for step in range(batch_step):
        lower = batch_size * step
        upper = batch_size + lower

        q, c = batch_slice(lower, upper, test_q, test_c)
        yield pad_seq(q), c


def AAP_generator(
    item_attr: pd.DataFrame, n_item: int, n_attr: int, batch_size: int
) -> Tuple[np.ndarray]:
    """
    for each item, generate indices of the item and ont-hot encodings of target attrbutes of the item

    a batch of item: [1], [2], [3]
    a batch of attrs: [1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]
    """
    item = item_attr.item_id.to_numpy()
    attr = item_attr.attr_list.apply(lambda x: [x]).to_numpy()

    one_hot = lambda d: one_hot_encoding(d, 1, n_attr)

    while True:
        item, attr = shuffle(n_item, item, attr)
        batch_step = int(np.ceil(n_item / batch_size))
        for step in range(batch_step):
            lower = batch_size * step
            upper = batch_size + lower

            i_b, a_b = batch_slice(lower, upper, item, attr)
            yield (i_b, one_hot(a_b))


def mip_train_data(user_item: pd.DataFrame, n_item: int) -> pd.DataFrame:
    """
    just add columns of "masked", "pos_targ", "neg_targ" by the function "make_mask"
    from "item_list" column
    """
    temp_df = user_item.assign(item_list=user_item.item_list.apply(lambda x: x[:-3]))
    temp_df[["masked", "pos_targ", "neg_targ"]] = temp_df.apply(
        lambda x: make_mask(x.item_list, n_item), axis=1, result_type="expand"
    )
    return temp_df.drop(columns=["user_id", "item_list"])


def MIP_generator(
    user_item: pd.DataFrame, n_user: int, n_item: int, max_item_len: int, batch_size: int
) -> Tuple[np.ndarray]:
    """
    for each user, generate randomly masked indices of interacted items and its target items

    a batch of masked items: [0, 0, 1, 0, 3]
    a batch of positive targets: [0, 0, 0, 2, 0]
    a batch of negative targets: [0, 0, 0, 99, 0]
    """
    pad_seq = lambda d: pad_sequences(d, max_item_len, padding="pre")
    while True:
        train = mip_train_data(
            user_item, n_item
        )  # initialize sampling for each epoch, but consume too much time..
        masked = train.masked.to_numpy()
        pos_targ = train.pos_targ.to_numpy()
        neg_targ = train.neg_targ.to_numpy()

        masked, pos_targ, neg_targ = shuffle(n_user, masked, pos_targ, neg_targ)
        batch_step = int(np.ceil(n_user / batch_size))
        for step in range(batch_step):
            lower = batch_size * step
            upper = batch_size + lower

            m_b, p_b, n_b = batch_slice(lower, upper, masked, pos_targ, neg_targ)
            yield tuple(map(pad_seq, (m_b, p_b, n_b)))


def map_train_data(user_item: pd.DataFrame, item_attr: pd.DataFrame) -> pd.DataFrame:
    """
    create user-attr interaction dataframe to pretrain

    item_list: [1, 2, 3]
    attr_list: [[11, 13, 24], [9, 10], [4, 9 ,10, 11, 12]]
    """
    temp_df = user_item.assign(item_list=user_item.item_list.apply(lambda x: x[:-3]))
    item2attr = {
        item_attr.at[idx, "item_id"]: item_attr.at[idx, "attr_list"] for idx in item_attr.index
    }
    user_attr = temp_df.assign(
        attr_list=lambda df: df.item_list.apply(lambda x: [item2attr[i] for i in x])
    )

    return user_attr.drop(columns=["user_id"])


def MAP_generator(
    user_item: pd.DataFrame,
    item_attr: pd.DataFrame,
    n_user: int,
    n_attr: int,
    max_item_len: int,
    batch_size: int,
) -> Tuple[np.ndarray]:
    """
    for each user, generate indices of interacted items and ont-hot encodings of target attrbutes of the items respectively

    a batch of mask padded items: [0, 0, 1, 2]
    a batch of one-hot encoded attrs: [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0]]
    """
    user_attr = map_train_data(user_item, item_attr)
    item = user_attr.item_list.to_numpy()
    attr = user_attr.attr_list.to_numpy()

    pad_seq = lambda d: pad_sequences(d, max_item_len, padding="pre")
    one_hot = lambda d: one_hot_encoding(d, max_item_len, n_attr)

    while True:
        item, attr = shuffle(n_user, item, attr)
        batch_step = int(np.ceil(n_user / batch_size))
        for step in range(batch_step):
            lower = batch_size * step
            upper = batch_size + lower
            i_b, a_b = batch_slice(lower, upper, item, attr)
            yield pad_seq(i_b), one_hot(a_b)


def sp_train_data(user_item: pd.DataFrame, n_item: int) -> pd.DataFrame:
    """
    just add columns of "masked", "pos_seg", "neg_seg" by the function "make_segment"
    from the "item_list" column
    """
    temp_df = user_item.assign(item_list=user_item.item_list.apply(lambda x: x[:-3]))
    temp_df[["masked", "pos_seg", "neg_seg"]] = temp_df.apply(
        lambda x: make_segment(x.item_list, n_item), axis=1, result_type="expand"
    )
    return temp_df.drop(columns=["user_id", "item_list"])


def SP_generator(
    user_item: pd.DataFrame, n_user: int, n_item: int, max_item_len: int, batch_size: int
) -> Tuple[np.ndarray]:
    """
    for each user, generate sequencial masked indices of interacted items and target items

    a batch of surr: [0, 1, 0, 0, 4]
    a batch of pos_seg: [0, 0, 2, 3, 0]
    a batch of neg_seg: [0, 0, 98, 99, 0]
    """
    pad_seq = lambda d: pad_sequences(d, max_item_len, padding="pre")
    while True:
        train = sp_train_data(
            user_item, n_item
        )  # initialize sampling for each epoch, but consume too much time..
        masked = train.masked.to_numpy()
        pos_seg = train.pos_seg.to_numpy()
        neg_seg = train.neg_seg.to_numpy()

        masked, pos_seg, neg_seg = shuffle(n_user, masked, pos_seg, neg_seg)
        batch_step = int(np.ceil(n_user / batch_size))
        for step in range(batch_step):
            lower = batch_size * step
            upper = batch_size + lower

            m_b, p_b, n_b = batch_slice(lower, upper, masked, pos_seg, neg_seg)
            yield tuple(map(pad_seq, (m_b, p_b, n_b)))
