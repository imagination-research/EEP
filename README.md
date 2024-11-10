# Efficient Expert Pruning for Sparse Mixture-of-Experts Models: Enhancing Performance and Reducing Inference Costs

This repository contains the official code for the paper [Efficient Expert Pruning for Sparse Mixture-of-Experts Models: Enhancing Performance and Reducing Inference Costs](http://arxiv.org/abs/2407.00945).

Currently, this repository is an initial version that supports only the evaluation of our pruning patterns. We provide our searched results on different datasets in [searched_coeff](searched_coeff). The implementation of evolutionary search will be released later.

This repository is heavily based on commit a00e57296fd0ebd63ccf47f78df54a68e8e551f9 of the [OpenCompass](https://github.com/open-compass/opencompass/tree/a00e57296fd0ebd63ccf47f78df54a68e8e551f9) codebase. Please note that since the OpenCompass codebase is frequently updated, the latest version of OpenCompass may not be compatible with our code.

If you find this repository or our paper useful, please cite
```
@article{liu2024efficient,
  title={Efficient expert pruning for sparse mixture-of-experts language models: Enhancing performance and reducing inference costs},
  author={Liu, Enshu and Zhu, Junyi and Lin, Zinan and Ning, Xuefei and Blaschko, Matthew B and Yan, Shengen and Dai, Guohao and Yang, Huazhong and Wang, Yu},
  journal={arXiv preprint arXiv:2407.00945},
  year={2024}
}
```

## Preparation

To install all necessary packages, run the following commands:
```sh
pip install ./transformers
pip install -r requirements.txt
```

## Inference

### EEP

Below are commands to reproduce our main results (Prune+Merge). Please replace "{DATASET}" with the name of your target dataset, and specify "{BUDGET}" as the number of experts. Available datasets include: wic, wsc, rte, boolq, cb, record, drop, squad (BUDGET options: 4 or 2 for these), winogrande, obqa, hellaswag and gsm8k (BUDGET option: 6 for these last four datasets). For some datasets, we use a random subset, with the indices uploaded in [data](data).

```
python opencompass/tasks/openicl_infer.py dataset_cfgs/mixtral-8x7b-instruct-v0.1_{DATASET}_eval.py --model_path /PATH/TO/YOUR/MIXTRAL-8X7B-INSTRUCT-V0.1/MODEL --coeff_path searched_coeff/{DATASET}_budget_{BUDGET}.pth --work_dir outputs/{DATASET}/EEP_budget_{BUDGET}
```
The inference and evaluation process are the same as in [OpenCompass](https://github.com/open-compass/opencompass/tree/a00e57296fd0ebd63ccf47f78df54a68e8e551f9). Generated answers and evaluated scores will be automatically saved in the specified working directory.

### Baselines

We also provide commands to run baseline methods for comparison with our EEP.

To conduct inference with the full model, please use: 
```
python opencompass/tasks/openicl_infer.py dataset_cfgs/mixtral-8x7b-instruct-v0.1_{DATASET}_eval.py --model_path /PATH/TO/YOUR/MIXTRAL-8X7B-INSTRUCT-V0.1/MODEL --work_dir outputs/{DATASET}/full_model
```

To reproduce the results of random pruning, please use: 
```
python opencompass/tasks/random_prune.py dataset_cfgs/mixtral-8x7b-instruct-v0.1_{DATASET}_eval.py --model_path /PATH/TO/YOUR/MIXTRAL-8X7B-INSTRUCT-V0.1/MODEL --budget {BUDGET} --work_dir outputs/{DATASET}/Random_budget_{BUDGET}
```