# S3Rec

- [S3-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization](https://arxiv.org/pdf/2008.07873.pdf)

## Library Info
- numpy == 1.19.2
- pandas == 1.1.3
- tensorflow == 2.4.0
- sklearn == 0.23.2
- fire
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

1. 현재 구현 결과는 논문의 결과보다 훨씬 좋지 않음
   - 논문 수식에 오류가 있어 논문 공식 구현을 참조해야 함
   - Loss를 구현하거나 negative sampling 하는 과정에서 차이가 있음

2. 논문에 기록된 것보다 훨씬 적은 실제 Beauty 데이터의 Attribute 개수
   - 논문 1,221개 vs 실제 데이터 637개
   - 이를 해결하고자 Amazon Beauty 데이터를 다운로드 받았으나, attribute에 해당하는 "category" 파트가 없었음

3. 나중에 개선해야할 점
   - 논문의 pytorch 구현 참고하여 구현할 것
   - argparse 써서 parameter 먹이기
   - 디자인 패턴 고려해서 코드 리팩토링 필요
   - tensorflow 공부 더해서 속도 측면에서 perfomance 올리기
