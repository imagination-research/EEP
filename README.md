# Efficient Expert Pruning for Sparse Mixture-of-Experts Models: Enhancing Performance and Reducing Inference Costs (NeurIPS 2024 Submission)

This is a simple repository to valid our results in Tab.1. We provide commands to reproduce the results of (1) full model, (2) random search and (3) EEP (Prune+Merge) in Tab.1.

This repository is based on commit a00e57296fd0ebd63ccf47f78df54a68e8e551f9 of the [open-compass](https://github.com/open-compass/opencompass/tree/a00e57296fd0ebd63ccf47f78df54a68e8e551f9) code base.

If your want to reproduce more results, please contact us by follow-up comment.

## Preparation

Please complete the following two steps before you begin to reproduce our results.

- Download Mixtral-8x7B-Instruct-v0.1 from the [huggingface](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)

- Make sure the version of your transformers package is 4.39.1 and locate its source code. Replace `/PATH/TO/TRANSFORMERS/transformers/models/mixtral/modeling_mixtral.py` with the new py file `./modeling_mixtral.py` in this repository. For example, my path of this file is `/path/to/my/anaconda/envs/my_env_name/lib/python3.10/site-packages/transformers/models/mixtral/modeling_mixtral.py`.

## Command

We first provide a quick command to valid the results of EEP (Prune+Merge) and the full model in Tab.1 on all datasets in one run. Please run:
```
python opencompass/tasks/openicl_reproduce.py dataset_cfgs/mixtral-8x7b-instruct-v0.1_boolq_eval.py --model_path /PATH/TO/YOUR/MIXTRAL-8X7B-INSTRUCT-V0.1/MODEL
```

If you want to separately validate results on each dataset, we provide commands for reproduction below. Please replace "{DATASET}" with the name of your target dataset. All available datasets are: wic, wsc, rte, boolq, cb, record, drop, squad. "{BUDGET}" can be specified as 4 or 2.

To reproduce the results of full model, please use: 
```
python opencompass/tasks/openicl_infer.py dataset_cfgs/mixtral-8x7b-instruct-v0.1_{DATASET}_eval.py --model_path /PATH/TO/YOUR/MIXTRAL-8X7B-INSTRUCT-V0.1/MODEL
```

To reproduce the results of random pruning, please use: 
```
python opencompass/tasks/random_prune.py dataset_cfgs/mixtral-8x7b-instruct-v0.1_{DATASET}_eval.py --model_path /PATH/TO/YOUR/MIXTRAL-8X7B-INSTRUCT-V0.1/MODEL --budget {BUDGET}
```

To reproduce the results of EEP, please use: 
```
python opencompass/tasks/openicl_infer.py dataset_cfgs/mixtral-8x7b-instruct-v0.1_{DATASET}_eval.py --model_path /PATH/TO/YOUR/MIXTRAL-8X7B-INSTRUCT-V0.1/MODEL --coeff_path searched_coeff/{DATASET}_budget_{BUDGET}.pth
```
