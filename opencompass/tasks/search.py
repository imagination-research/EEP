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

import sys
sys.path.insert(0, ".")

from opencompass.registry import (ICL_INFERENCERS, ICL_PROMPT_TEMPLATES,
                                  ICL_RETRIEVERS, TASKS)
from opencompass.tasks.base import BaseTask
from opencompass.utils import (build_dataset_from_cfg, build_model_from_cfg,
                               get_infer_output_path, get_logger,
                               task_abbr_from_cfg)

from opencompass.models.huggingface import HuggingFaceCausalLM

from opencompass.tasks.openicl_eval import OpenICLEvalTask
from opencompass.tasks.openicl_infer import OpenICLInferTask
from opencompass.tasks.search_utils import *

            
def set_router_transfer(model, coeff, target_expert_num):
    for layer_id in range(len(model.model.model.layers)):
        for key in coeff.keys():
            if "transfer" in key:
                if pattern_in(f"transfer.{layer_id}", key):
                    break
        else:
            key = "others_transfer"
        # print(f"layer_id {layer_id}, key {key}")
        assert target_expert_num == coeff[key].shape[0]
        if hasattr(model.model.model.layers[layer_id], "block_sparse_moe"):
            module = model.model.model.layers[layer_id].block_sparse_moe
        elif hasattr(model.model.model.layers[layer_id], "mlp"):
            module = model.model.model.layers[layer_id].mlp
        if not hasattr(module, "gate"):
            continue
        module.expert_eliminate = True
        module.router_transfer = coeff[key].to(module.gate.weight) if isinstance(coeff[key], torch.Tensor) else torch.tensor(coeff[key]).to(module.gate.weight)
        module.num_experts = target_expert_num
        module.gate.expert_eliminate = module.expert_eliminate
        module.gate.router_transfer = module.router_transfer
        module.gate.num_experts = module.num_experts
    
def merge_weights(model, coeff, expert_weights_dict, ref_state_dict, target_num_expert, prefix="block_sparse_moe.experts"):
    print(f"Begin weight merging...")
    # print(f"coeff: {coeff}")
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
    
def set_router_transfer_and_prob(cfg):
    for key in cfg["weight_group"]:
        if "transfer" in key:
            print(f"The cfg has already had transfer groups!")
            return cfg
    transfer_key = []
    change_prob_weight = []
    change_prob_transfer = []
    for key in cfg["weight_group"]:
        if key != "others":
            transfer_key.append(key.replace("layers", "transfer"))
        else:
            transfer_key.append("others_transfer")
        change_prob_weight.append(cfg["weight_change_prob"])
        change_prob_transfer.append(cfg["transfer_change_prob"])
    cfg["weight_group"] = cfg["weight_group"] + transfer_key
    cfg["group_change_prob"] = change_prob_weight + change_prob_transfer
    return cfg

def set_all_weights_group(layer_num, cfg, group_num=None):
    cfg["weight_group"] = []
    if group_num is None:
        group_num = layer_num
        cfg["weight_group"] = [f"layer.{idx}" for idx in range(group_num - 1)]
    
    group_start = 0
    for i in range(group_num - 1):
        group_end = group_start + layer_num // group_num - 1
        cfg["weight_group"].append(f"layer.[{group_start}-{group_end}]")
        group_start = group_end + 1
    cfg["weight_group"].append("others")
    return cfg

def parse_args():
    parser = argparse.ArgumentParser(description='Model Inferencer')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('--model_path', help='path to Mixtral-8x7B-Instruct-v0.1 huggingface model', required=True)
    parser.add_argument('--eval_cfg', help='Evaluation config file path', default="eval")
    parser.add_argument('--dev_cfg', help='Validation config file path', default=None)
    parser.add_argument('--dev_num', help='Number of coefficients to test on the dev set', type=int, default=20)
    parser.add_argument('--search_config', help='Search config file path')
    parser.add_argument('--data_path', help='Path to save all evaluated coefficients')
    parser.add_argument('--split', default="0")
    parser.add_argument('--seed', help="Random seed", type=int, default=11450)
    parser.add_argument('--discrete_iter', help="Iteration of the pruning stage", type=int, default=None)
    parser.add_argument('--total_iter', help="Total iteration of both stages", type=int, default=None)
    parser.add_argument('--metric', help="Type of metric", type=str, default=None)
    parser.add_argument('--budget', help="The number of kept experts", type=int, default=None)
    parser.add_argument('--load_expert_gpu', action="store_true", help="By using extra GPU memory, the need to move the original expert weights between GPU and CPU is eliminated.")
    parser.add_argument('--active_num', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    args = parser.parse_args()
    return args

def main():
    # get args and model config
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg["models"][0]["path"] = args.model_path
    cfg["models"][0]["tokenizer_path"] = args.model_path
    
    # get search config
    with open(args.search_config, "r") as f:
        search_cfg = yaml.safe_load(f)
        search_cfg["budget"] = args.budget
    if args.discrete_iter is not None:
        search_cfg["discrete_iter"] = args.discrete_iter
    if args.total_iter is not None:
        search_cfg["total_iter"] = args.total_iter
    if args.metric is not None:
        search_cfg["metric"]["type"] = args.metric
    
    # path preparation  
    save_path = os.path.join(args.data_path, args.split)
    os.makedirs(save_path, exist_ok=True)

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if args.batch_size is not None:
        cfg["models"][0]["batch_size"] = args.batch_size
    if args.active_num is not None:
        cfg["models"][0]["num_experts_per_tok"] = args.active_num
        
    # set cfg
    cfg["work_dir"] = os.path.join(save_path, "temp_results")
    cfg_save = copy.deepcopy(cfg)

    # set inferencer and evaler
    inferencer = OpenICLInferTask(cfg)
    evaler = OpenICLEvalTask(cfg)
    
    # init model
    model_cfg = inferencer.model_cfgs[0] # suppose that we have only one MoE model to merge
    model = build_model_from_cfg(model_cfg)
    
    # get weight of experts
    model_state_dict = model.model.state_dict()
    assert isinstance(model, HuggingFaceCausalLM)
    if hasattr(model.model.model.layers[0], "block_sparse_moe"): # for Mixtral models
        num_expert = model.model.model.layers[0].block_sparse_moe.num_experts
        prefix = "block_sparse_moe.experts"
    elif hasattr(model.model.model.layers[0], "mlp"): # for Qwen, DeepSeek models
        if not hasattr(model.model.model.layers[0].mlp, "num_experts"):
            model.model.model.layers[0].mlp.num_experts = len(model.model.model.layers[1].mlp.experts)
        num_expert = model.model.model.layers[0].mlp.num_experts
        prefix = "mlp.experts"
    expert_weights_dict = {}
    for key in model_state_dict.keys():
        expert_weights_dict[key] = copy.deepcopy(model_state_dict[key].cpu()) if not args.load_expert_gpu else copy.deepcopy(model_state_dict[key])
    
    # TODO: Reduce the number of experts in the model to the specified budget in order to lower GPU memory consumption.
        
    # evolutionary search
    init_done = False
    discrete_eval = False
    os.makedirs(os.path.join(args.data_path, "init"), exist_ok=True)
    metric_cfg = search_cfg["metric"][search_cfg["metric"]["type"]] # TODO: args.metric_type
    search_cfg = set_all_weights_group(len(model.model.model.layers), search_cfg)
    search_cfg = set_router_transfer_and_prob(search_cfg)
    
    def eval_results(name):
        if args.dev_cfg is not None:
            dev_cfg_path = args.config.replace("train.py", f"{args.dev_cfg}.py")
            dev_cfg = Config.fromfile(dev_cfg_path)
            dev_cfg["work_dir"] = os.path.join(save_path, f"results_dev_{name}")
            if args.batch_size is not None:
                dev_cfg["models"][0]["batch_size"] = args.batch_size
            dev_cfg_save = copy.deepcopy(dev_cfg)
            dev_inferencer = OpenICLInferTask(dev_cfg)
            dev_evaler = OpenICLEvalTask(dev_cfg)
            population = get_all_data(args.data_path, metric_cfg)
            dev_metrics = []
            for i in range(args.dev_num):
                print(f"Evaluating coeff: {population[i]}")
                # remove temp results
                if os.path.exists(dev_cfg["work_dir"]):
                    shutil.rmtree(dev_cfg["work_dir"])
                coeff = population[i][0]
                with torch.no_grad():
                    merge_weights(model, coeff, expert_weights_dict, model_state_dict, target_num_expert=args.budget, prefix=prefix)
                    dev_inferencer.run(model=model)
                results = dev_evaler.run()
                metric = get_metric(results, search_cfg)
                print(f"Dev metric: {metric}")
                dev_metrics.append(metric)
                # reset inferencer and evaler
                dev_cfg = copy.deepcopy(dev_cfg_save)
                dev_inferencer = OpenICLInferTask(dev_cfg)
                dev_evaler = OpenICLEvalTask(dev_cfg)
            select_index = np.argmax(dev_metrics)
            print(f"Choose {select_index}-th coefficient. The dev metric is {dev_metrics[select_index]}")
        else:
            select_index = 0
        
        # evaluate the search result
        eval_cfg_path = args.config.replace("train.py", f"{args.eval_cfg}.py")
        eval_cfg = Config.fromfile(eval_cfg_path)
        eval_cfg["work_dir"] = os.path.join(save_path, f"results_eval_{name}")
        if args.batch_size is not None:
            eval_cfg["models"][0]["batch_size"] = args.batch_size
        eval_inferencer = OpenICLInferTask(eval_cfg)
        eval_evaler = OpenICLEvalTask(eval_cfg)
        population = get_all_data(args.data_path, metric_cfg)
        print(f"Evaluating final coeff: {population[select_index]}")
        final_coeff = population[select_index][0]
        # remove temp results
        if os.path.exists(eval_cfg["work_dir"]):
            shutil.rmtree(eval_cfg["work_dir"])
        with torch.no_grad():
            merge_weights(model, final_coeff, expert_weights_dict, model_state_dict, target_num_expert=args.budget, prefix=prefix)
            eval_inferencer.run(model=model)
        results = eval_evaler.run()
        metric = get_metric(results, search_cfg)
        print(f"{name} evaluation metric: {metric}")
        
    while(1):
        
        # remove temp results
        if os.path.exists(cfg["work_dir"]):
            shutil.rmtree(cfg["work_dir"])
            
        # get all evaluated coefficients and sort them
        population = get_all_data(args.data_path, metric_cfg)

        # get coefficients
        if not init_done: # get init 
            all_init_path = [os.path.join(args.data_path, "init", f) for f in os.listdir(os.path.join(args.data_path, "init"))]
            coeff, cur_save_path = get_init(search_cfg, args.data_path, num_expert=num_expert, all_init_path=all_init_path)
            if coeff == "continue enumeration":
                continue
            
        if init_done or (coeff is None and cur_save_path is None):
            init_done = True
            save_name = str(len(os.listdir(save_path)))
            
            # whether to change each group
            assert len(search_cfg["weight_group"]) == len(search_cfg["group_change_prob"])
            while(1):
                group_change = {}
                for i in range(len(search_cfg["weight_group"])):
                    group_change[search_cfg["weight_group"][i]] = random.uniform(0, 1) < search_cfg["group_change_prob"][i]
                if 1 in list(group_change.values()):
                   break
            change_name = [k for k in group_change.keys() if group_change[k]]

            # crossover
            if random.uniform(0, 1) < search_cfg["crossover"]["prob"] and len(population) > search_cfg["discrete_iter"]:
                coeff = crossover(search_cfg, population, group_change)
            else:
                coeff = None
            check(search_cfg, coeff)
            
            # mutate
            if coeff is None:
                if search_cfg["mutate"]["parents_select_type"] == "ranking":
                    parents = population[min(len(population) - 1, random.randint(0, search_cfg["mutate"]["parents_rank"]))]
                    print(f"The score of parents {parents[1]}")
                    coeff = parents[0]
                elif search_cfg["mutate"]["parents_select_type"] == "absolute":
                    best = population[0][1][1]
                    threshold = search_cfg["mutate"]["parents_threshold"]
                    for i in range(1, len(population)):
                        if metric_cfg["better"] == "larger":
                            if best - population[i][1][1] > threshold and best - population[i - 1][1][1] <= threshold:
                                break
                        else:
                            raise NotImplementedError
                    i = max(i, search_cfg["mutate"]["parents_rank"])
                    parents = population[min(len(population) - 1, random.randint(0, i))]
                    print(f"The score of parents {parents[1]}")
                    coeff = parents[0]
                else:
                    raise NotImplementedError

            if len(population) > search_cfg["discrete_iter"]:
                if not discrete_eval:
                    eval_results(name="prune")
                discrete_eval = True
                # TODO: only select the one with best val acc
                coeff = mutate(search_cfg, coeff, group_change)
            else:
                coeff = discrete_mutate(search_cfg, coeff, group_change)
            
            # normalize
            if random.uniform(0, 1) < search_cfg["norm_prob"]:
                for key in coeff.keys():
                    if not "scaling" in key and not "transfer" in key:
                        for i in range(len(coeff[key])):
                            if isinstance(coeff[key], np.ndarray):
                                coeff[key][i] = coeff[key][i] / np.sum(coeff[key][i])
                            elif isinstance(coeff[key], torch.Tensor):
                                coeff[key][i] = coeff[key][i] / torch.sum(coeff[key][i])
            
            save_name += ".pth"
            cur_save_path = os.path.join(save_path, save_name)
     
        
        # check whether the current coefficient is correct
        check(search_cfg, coeff) 
        # print("final check done!")


        # evaluation
        # print(f"evaluating coeff {coeff}")
        with torch.no_grad():
            merge_weights(model, coeff, expert_weights_dict, model_state_dict, target_num_expert=args.budget, prefix=prefix)
            inferencer.run(model=model)
            
        results = evaler.run()
        assert len(results) == 1
        
        # get metric
        metric = get_metric(results, search_cfg)
        
        # save evaluated coefficient and its result
        if "_place_holder" in str(cur_save_path):
            os.remove(cur_save_path)
            cur_save_path = str(cur_save_path).replace("_place_holder", ".pth")
        try:
            torch.save((coeff, metric), cur_save_path)
            print(f"save {(coeff, metric)} to {cur_save_path} success, continue...")
        except:
            print(f"save {(coeff, metric)} to {cur_save_path} fail, continue...")
            import ipdb; ipdb.set_trace()
        
        # reset inferencer and evaler
        cfg = copy.deepcopy(cfg_save)
        inferencer = OpenICLInferTask(cfg)
        evaler = OpenICLEvalTask(cfg)
        
        if len(population) > search_cfg["total_iter"]:
            break
        
    eval_results(name="merge")

if __name__ == '__main__':
    main()
