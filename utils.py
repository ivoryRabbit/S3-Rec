import json
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import MultiLabelBinarizer


def load_user_item(path: str):
    """
    user_id: 1, 2, 3, ...
    item_list: [1, 2, 3], [4, 5], [2, 3, 4], ...
    """
    user_item = []
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            line = list(map(int, line.split()))
            user, item = line[0], line[1:]
            user_item.append({"user_id": user, "item_list": item})
    f.close()

    user_item = pd.DataFrame(user_item)
    user_item = user_item.sort_values(by="user_id")
    return user_item.reset_index(drop=True)


def load_item_attr(path: str):
    """
    item_id: 1, 2, 3, ...
    attr_list: [1, 2, 3], [4, 5], [2, 3, 4], ...
    """
    with open(path, encoding="utf-8") as f:
        item_attr = json.load(f)
    f.close()
    item_attr = [{"item_id": int(item), "attr_list": item_attr[item]} for item in item_attr.keys()]
    item_attr = pd.DataFrame(item_attr)
    item_attr = item_attr.sort_values(by="item_id")
    return item_attr.reset_index(drop=True)


def neg_sampling(pos_list: list, total_size: int, ns: int):
    """
    total_size = the size of candidates for sampling
    ns = the number of negative samples per a positive sample
    """
    neg_list = []
    while len(neg_list) < ns:
        neg = np.random.randint(1, total_size + 1)
        while (neg in pos_list) & (neg in neg_list):
            neg = np.random.randint(1, total_size + 1)
        neg_list.append(neg)
    return neg_list


def shuffle(n_user: int, *data: np.ndarray):
    # shuffle arrays
    seed = np.random.permutation(n_user)
    return [d[seed] for d in data]


def batch_slice(lower: int, upper: int, *data: np.ndarray):
    # slice arrays
    return [d[lower:upper] for d in data]


def one_hot_encoding(attr_lists: np.ndarray, max_item_len: int, n_attr: int):
    """
    get multi-labeled one-hot encodings from indices and pad zero arrays
    if max_attr_len = 3 and n_attr = 4, then
    [[[1, 3], [2], [1]],
     [[1, 2], [2, 4]]]
    ->
     [[[1, 0, 1, 0],
       [0, 1, 0, 0],
       [1, 0, 0, 0]],
       [0, 0, 0, 0],
       [1, 1, 0, 0],
       [0, 1, 0, 1]]]
    """
    MLB = MultiLabelBinarizer(np.arange(1, n_attr + 1))
    res = []
    for attr_list in attr_lists:
        encoded = MLB.fit_transform(attr_list)[
            -max_item_len:
        ]  # discard attributes corresponding discarded items
        if max_item_len > 1:
            encoded = np.pad(encoded, ((max_item_len - len(encoded), 0), (0, 0)))
        res.append(encoded)
    return np.squeeze(np.stack(res))


def make_mask(item_list: list, n_item: int):
    """
    if item_list = [1, 2, 3, 4], then
    masked = [1, 2, 0, 4]
    pos_targ = [0, 0, 3, 0]
    neg_targ = [0, 0, 99, 0]
    """
    masked = np.array(item_list)
    mask_len = int(np.ceil(len(item_list) / 5))  # proportion = 0.2
    mask_idx = np.random.choice(range(len(masked)), size=mask_len)

    pos_targ = np.zeros(shape=(len(item_list)), dtype=np.int32)
    pos_targ[mask_idx] = masked[mask_idx]

    neg = neg_sampling(item_list, n_item, 1)[:mask_len]
    neg_targ = np.zeros(shape=(len(item_list)), dtype=np.int32)
    neg_targ[mask_idx] = neg

    masked[mask_idx] = 0
    return masked, pos_targ, neg_targ


def make_segment(item_list: list, n_item: int):
    """
    if item_list = [1, 2, 3, 4], then
    masked = [1, 0, 0, 4]
    pos_seg = [0, 2, 3, 0]
    neg_seg = [0, 98, 99, 0]
    """
    masked = np.array(item_list)
    seg_len = np.random.randint(1, len(item_list) // 2 + 1)
    start = np.random.randint(0, len(item_list) // 2)
    end = start + seg_len

    pos_seg = np.zeros(shape=(len(item_list)), dtype=np.int32)
    pos_seg[start:end] = masked[start:end]

    neg = neg_sampling(item_list, n_item, ns=end - start)
    neg_seg = np.zeros(shape=(len(item_list)), dtype=np.int32)
    neg_seg[start:end] = neg

    masked[start:end] = 0
    return masked, pos_seg, neg_seg
