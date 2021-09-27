# SeTransE: Enhancing Semantic Awareness in Knowledge Graph Embedding
This is the code of paper **Enhancing Semantic Awareness in Knowledge Graph Embedding.** 

## Dependencies
- Python 3.6+
- [PyTorch](http://pytorch.org/) 1.0+


## Results
The results of **SeTransE** on **WN18RR**, **FB15k**, **FB15k-237** and **YAGO3-10** are as follows.

### WN18RR
| MRR |  HITS@1 | HITS@3 | HITS@10 |
|:----------:|:----------:|:----------:|:----------:|
| 0.456 | 0.387 | 0.436 | 0.526 |

### FB15k
| MRR |  HITS@1 | HITS@3 | HITS@10 |
|:----------:|:----------:|:----------:|:----------:|
| 0.812 | 0.766 | 0.874 | 0.876 |

### FB15k-237
| MRR |  HITS@1 | HITS@3 | HITS@10 |
|:----------:|:----------:|:----------:|:----------:|
| 0.317 | 0.281 | 0.332 | 0.471 |

### YAGO3-10
| MRR |  HITS@1 | HITS@3 | HITS@10 |
|:----------:|:----------:|:----------:|:----------:|
| 0.470 | 0.377 | 0.486 | 0.564 |
 
## Running the code 

### Usage
```
bash runs.sh {train | test} {WN18rr | WN18 | FB15k | FB15k-237 | YAGO3-10} <train_batch_size>\
<embedding_dim><margin><learning_rate><epochs>
```
- `{ | }`: Mutually exclusive items. Choose one from them.
- `< >`: Placeholder for which you must supply a value.

**Remark**: `[modulus_weight]` and `[phase_weight]` are available only for the `HAKE` model.

To reproduce the results of SeTransE, run the following commands.

### HAKE
```
# WN18rr
bash runs.sh train WN18rr 500 1.0 0.5 0.00007 50000
bash runs.sh test WN18rr
# FB15k-237
bash runs.sh train FB15k-237  1024 1.0 0.00005 100000
bash runs.sh test FB15k-237
# YAGO3-10
bash runs.sh train YAGO3-10 1024 1.0 0.0002 180000
bahs runs.sh test YAGO3-10
# WN18
bash runs.sh train WN18 500 1.0 0.5 0.00005 80000
bash runs.sh test WN18
# FB15k
bash runs.sh train FB15k  1024 1.0 0.00007 150000
bash runs.sh test FB15k
```


