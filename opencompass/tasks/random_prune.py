import sys
sys.path.insert(0, ".")

import argparse
import os.path as osp
import os
import random
import copy
import yaml
import time
import shutil
import numpy as np
import torch
from shutil import which
from typing import Any

from mmengine.config import Config, ConfigDict
from mmengine.utils import mkdir_or_exist

from opencompass.registry import (ICL_INFERENCERS, ICL_PROMPT_TEMPLATES,
                                  ICL_RETRIEVERS, TASKS)
from opencompass.tasks.base import BaseTask
from opencompass.utils import (build_dataset_from_cfg, build_model_from_cfg,
                               get_infer_output_path, get_logger,
                               task_abbr_from_cfg)

from opencompass.models.huggingface import HuggingFaceCausalLM

from opencompass.tasks.openicl_eval import OpenICLEvalTask
from opencompass.tasks.openicl_infer import OpenICLInferTask
from opencompass.tasks.merge_utils import *

from transformers.models.mixtral import MixtralForCausalLM
            
def set_router_transfer(model, coeff, target_expert_num):
    for layer_id in range(len(model.model.model.layers)):
        for key in coeff.keys():
            if "transfer" in key:
                if pattern_in(f"transfer.{layer_id}", key):
                    break
        else:
            key = "others_transfer"
        print(f"layer_id {layer_id}, key {key}")
        assert target_expert_num == coeff[key].shape[0]
        if hasattr(model.model.model.layers[layer_id], "block_sparse_moe"):
            module = model.model.model.layers[layer_id].block_sparse_moe
        elif hasattr(model.model.model.layers[layer_id], "mlp"):
            module = model.model.model.layers[layer_id].mlp
        module.expert_eliminate = True
        module.router_transfer = coeff[key].to(module.gate.weight) if isinstance(coeff[key], torch.Tensor) else torch.tensor(coeff[key]).to(module.gate.weight)
        module.num_experts = target_expert_num
    
def merge_weights(model, coeff, expert_weights_dict, ref_state_dict, target_num_expert, prefix="block_sparse_moe.experts"):
    for key in ref_state_dict.keys():
        if prefix in key:
            for group_name in coeff.keys():
                if pattern_in(key, group_name):
                    break
            else:
                group_name = "others"
            group_coeff = coeff[group_name]
            cur_id = get_cur_id(key, "experts")
            if cur_id < target_num_expert: # only experts which index is smaller than the target expert num should be merged
                cur_weight = None
                for i in range(len(group_coeff[cur_id])):
                    if cur_weight == None:
                        cur_weight = expert_weights_dict[replace_cur_id(key, cur_id, i, "experts")] * group_coeff[cur_id][i]
                    else:
                        if group_coeff[cur_id][i] != 0:
                            cur_weight += expert_weights_dict[replace_cur_id(key, cur_id, i, "experts")] * group_coeff[cur_id][i]
                ref_state_dict[key] = cur_weight
    
    model.model.load_state_dict(ref_state_dict)
    set_router_transfer(model, coeff, target_num_expert)

def get_expert_eliminate(num_expert=8, eliminate_expert=1):
    keep_experts = sorted(random.sample(range(num_expert), num_expert - eliminate_expert))
    full_matrix = torch.zeros(num_expert, num_expert)
    for i in range(num_expert):
        full_matrix[i, i] = 1.0
    group_matrix = full_matrix[keep_experts]
    group_coeff = group_matrix
    
    return group_coeff

def get_random_discrete_coefficient(num_expert, eliminate_expert, weight_group):
    coeff = {}
    for key in weight_group:
        if not "scaling" in key and not "transfer" in key:
            group_coeff = get_expert_eliminate(num_expert=num_expert, eliminate_expert=eliminate_expert)
            coeff[key] = group_coeff
            trans_key = key.replace("layers", "transfer") if not "others" in key else "others_transfer"
            coeff[trans_key] = copy.deepcopy(group_coeff)
    
    return coeff
    
def parse_args():
    parser = argparse.ArgumentParser(description='Model Inferencer')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('--budget', help="the number of kept experts", type=int, default=4)
    parser.add_argument('--weight_group_num', help="the number of weight group", type=int, default=4)
    parser.add_argument('--random_num', help="number of random pruning pattern", type=int, default=30)
    parser.add_argument('--model_path', help='path to Mixtral-8x7B-Instruct-v0.1 huggingface model', required=True)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg["models"][0]["path"] = args.model_path
    cfg["models"][0]["tokenizer_path"] = args.model_path
    cfg_save = copy.deepcopy(cfg)
    
    inferencer = OpenICLInferTask(cfg)
    evaler = OpenICLEvalTask(cfg_save)
    
    model_cfg = inferencer.model_cfgs[0] # suppose that we have only one MoE model to merge
    model = build_model_from_cfg(model_cfg)

    # get weight of experts
    model_state_dict = model.model.state_dict()
    if isinstance(model, HuggingFaceCausalLM):
        if hasattr(model.model.model.layers[0], "block_sparse_moe"):
            num_expert = model.model.model.layers[0].block_sparse_moe.num_experts
        elif hasattr(model.model.model.layers[0], "mlp"):
            num_expert = model.model.model.layers[0].mlp.num_experts
    expert_weights_dict = {}
    for key in model_state_dict.keys():
        if "block_sparse_moe.experts" in key:
            expert_weights_dict[key] = copy.deepcopy(model_state_dict[key].cpu())
            prefix = "block_sparse_moe.experts"
        elif "mlp.experts" in key:
            expert_weights_dict[key] = copy.deepcopy(model_state_dict[key].cpu())
            prefix = "mlp.experts"
    
    model_state_dict = model.model.state_dict()
    torch.cuda.empty_cache()
    
    if args.weight_group_num == 4:
        weight_group = ["layers.[0-7]", "layers.[8-15]", "layers.[16-23]", "others"]
    elif args.weight_group_num == 32:
        weight_group = ['layers.0', 'layers.1', 'layers.2', 'layers.3', 'layers.4', 'layers.5', 'layers.6', 'layers.7', 
               'layers.8', 'layers.9', 'layers.10', 'layers.11', 'layers.12', 'layers.13', 'layers.14', 'layers.15', 
               'layers.16', 'layers.17', 'layers.18', 'layers.19', 'layers.20', 'layers.21', 'layers.22', 'layers.23', 
               'layers.24', 'layers.25', 'layers.26', 'layers.27', 'layers.28', 'layers.29', 'layers.30', 'others']
    else:
        raise NotImplementedError("Currently only 4 groups and 32 groups of weights are supported!")
    
    all_random_perf = []
    for _ in range(args.random_num):
        cfg = Config.fromfile(args.config)
        cfg["models"][0]["path"] = args.model_path
        cfg["models"][0]["tokenizer_path"] = args.model_path
        cfg["work_dir"] = cfg["work_dir"] + f"random"
        if os.path.exists(cfg["work_dir"]):
            shutil.rmtree(cfg["work_dir"])
            
        coeff = get_random_discrete_coefficient(num_expert, num_expert - args.budget, weight_group)
        
        merge_weights(model, coeff, expert_weights_dict, model_state_dict, target_num_expert=args.budget, prefix=prefix)
        cfg_save = copy.deepcopy(cfg)
        start_time = time.time()
        inferencer = OpenICLInferTask(cfg)
        evaler = OpenICLEvalTask(cfg_save)
        inferencer.run(model=model)
        end_time = time.time()
        get_logger().info(f'time elapsed: {end_time - start_time:.2f}s')
        result = evaler.run()
        all_random_perf.append(list(result[0][0].values())[0])
        
    print(f"Mean accuracy: {sum(all_random_perf) / len(all_random_perf)}")

if __name__ == '__main__':
    main()
