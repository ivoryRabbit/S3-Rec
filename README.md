# S3Rec
---

## Paper
- [S3-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization](https://arxiv.org/pdf/2008.07873.pdf)

## Library Version Info
- numpy == 1.19.2
- pandas == 1.1.3
- tensorflow == 2.4.0
- sklearn == 0.23.2
- fire (argparse로 바꿀 예정)
- tqdm

## How to Run

### 1. Pretrain
~~~
$> python pretrains.py run\
        --user_item_fname=data\Beauty.txt \
        --item_attr_fname=data\Beauty_item2attributes.json \
        --verbose=1
~~~

### 2. Finetune
~~~
$> python main.py run\
        --user_item_fname=data\Beauty.txt \
        --use_pretrained=True \
        --verbose=1
~~~

### 3. Evaluation

~~~
$> python evals.py run\
        --user_item_fname=data\Beauty.txt \
        --use_pretrained=True
~~~

### (extra) Non Finetune
~~~
$> python main.py run\
        --user_item_fname=data\Beauty.txt \
        --use_pretrained=False \
        --verbose=1
~~~

~~~
$> python evals.py run\
        --user_item_fname=data\Beauty.txt \
        --use_pretrained=False
~~~

## Results

| metrics | paper score | my score |
|:---     | ---:| ---:|
| HR@1    | 0.2192 | 0.1320 |
| HR@5    | 0.4502 | 0.3222 |
| HR@10   | 0.5506 | 0.4297 |
| NDCG@5  | 0.3407 | 0.2301 |
| NDCG@10 | 0.3732 | 0.2648 |
| MRR     | 0.3340 | 0.2329 |

## 고찰

실험 결과 논문보다 훨씬 결과가 좋지 않습니다. 실제 구현과는 다음과 같은 차이가 있을거라 생각합니다.

1. 논문의 실제 구현과 본 구현에서의 차이
   - 제 구현이 미흡하거나 착오가 있을 가능성이 있습니다.
   - Loss를 구현하거나 negative sampling 하는 과정이 다른 것으로 보입니다.

2. 논문에 기록된 것보다 훨씬 적은 실제 Beauty 데이터의 Attribute 개수
   - 논문 1,221개 vs 실제 데이터 637개
   - 이를 해결하고자 Amazon Beauty 데이터를 다운로드 받았으나, 이곳에서는 attribute에 해당하는 "category" 파트가 없었습니다.
