import sys
sys.path.insert(0, ".")

import argparse
import os.path as osp
import random
import time
import copy
import torch
import os
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

from opencompass.tasks.openicl_eval import OpenICLEvalTask
from opencompass.models.huggingface import HuggingFaceCausalLM
from opencompass.tasks.merge_utils import *

from transformers.models.mixtral import modeling_mixtral

from datasets import load_dataset, concatenate_datasets

@TASKS.register_module(force=(__name__ == '__main__'))  # A hack for script run
class OpenICLInferTask(BaseTask):
    """OpenICL Inference Task.

    This task is used to run the inference process.
    """

    name_prefix = 'OpenICLInfer'
    log_subdir = 'logs/infer'
    output_subdir = 'predictions'

    def __init__(self, cfg: ConfigDict):
        super().__init__(cfg)
        run_cfg = self.model_cfgs[0].get('run_cfg', {})
        self.num_gpus = run_cfg.get('num_gpus', 0)
        self.num_procs = run_cfg.get('num_procs', 1)
        self.concat_dataset = self.dataset_cfgs[0][0].pop('concat_dataset', False)
        self.logger = get_logger()

    def get_command(self, cfg_path, template):
        """Get the command template for the task.

        Args:
            cfg_path (str): The path to the config file of the task.
            template (str): The template which have '{task_cmd}' to format
                the command.
        """
        script_path = __file__
        backend_keys = ['VLLM', 'Lmdeploy']
        use_backend = any(
            key in str(self.model_cfgs[0].get('type', ''))
            or key in str(self.model_cfgs[0].get('llm', {}).get('type', ''))
            for key in backend_keys)
        if self.num_gpus > 0 and not use_backend:
            port = random.randint(12000, 32000)
            command = (f'torchrun --master_port={port} '
                       f'--nproc_per_node {self.num_procs} '
                       f'{script_path} {cfg_path}')
        else:
            python = 'python3' if which('python3') else 'python'
            command = f'{python} {script_path} {cfg_path}'

        return template.format(task_cmd=command)
    
    def reset_config(self, model_cfgs, dataset_cfgs):
        self.model_cfgs = copy.deepcopy(model_cfgs)
        self.dataset_cfgs = copy.deepcopy(dataset_cfgs)

    def run(self, model=None):
        self.logger.info(f'Task {task_abbr_from_cfg(self.cfg)}')
        concat_dataset = self.concat_dataset
        self.logger.info(f'Concat dataset: {concat_dataset}')
        if model is not None:
            assert len(self.model_cfgs) == 1 # currently we do not support merge several models at the same time
        for model_cfg, dataset_cfgs in zip(self.model_cfgs, self.dataset_cfgs):
            self.max_out_len = model_cfg.get('max_out_len', None)
            self.batch_size = model_cfg.get('batch_size', None)
            self.min_out_len = model_cfg.get('min_out_len', None)
            if model is not None:
                self.model = model
            else:
                self.model = build_model_from_cfg(model_cfg)

            if not concat_dataset:
                for dataset_cfg in dataset_cfgs:
                    self.model_cfg = model_cfg
                    self.dataset_cfg = dataset_cfg
                    self.infer_cfg = self.dataset_cfg['infer_cfg']
                    self.dataset = build_dataset_from_cfg(self.dataset_cfg)
                    self.sub_cfg = {
                        'models': [self.model_cfg],
                        'datasets': [[self.dataset_cfg]],
                    }
                    out_path = get_infer_output_path(
                        self.model_cfg, self.dataset_cfg,
                        osp.join(self.work_dir, 'predictions'))
                    if osp.exists(out_path):
                        continue
                    self._inference()
            else:
                self.dataset = None
                self.per_dataset_len = []
                self.per_dataset_name = []
                self.infer_cfgs = []
                for dataset_cfg in dataset_cfgs:
                    tmp_dataset = build_dataset_from_cfg(dataset_cfg)
                    if self.dataset is None:
                        self.dataset = tmp_dataset
                    else:
                        for key in self.dataset.dataset.keys():
                            self.dataset.dataset[key] = concatenate_datasets([self.dataset.dataset[key], tmp_dataset.dataset[key]])
                    self.per_dataset_len.append(len(tmp_dataset.dataset["test"]))
                    self.per_dataset_name.append(dataset_cfg["abbr"])
                    self.infer_cfgs.append(dataset_cfg['infer_cfg'])
                
                self.model_cfg = model_cfg
                self.dataset_cfg = copy.deepcopy(dataset_cfgs[0])
                self.dataset_cfg["abbr"] = "whole_dataset"
                self.infer_cfg = self.dataset_cfg['infer_cfg']
                self.sub_cfg = {
                    'models': [self.model_cfg],
                    'datasets': [[self.dataset_cfg]],
                }
                out_path = get_infer_output_path(
                    self.model_cfg, self.dataset_cfg,
                    osp.join(self.work_dir, 'predictions'))
                if osp.exists(out_path):
                    continue
                self._inference()
                

    def _inference(self):
        self.logger.info(
            f'Start inferencing {task_abbr_from_cfg(self.sub_cfg)}')

        assert hasattr(self.infer_cfg, 'ice_template') or hasattr(self.infer_cfg, 'prompt_template'), \
            'Both ice_template and prompt_template cannot be None simultaneously.'  # noqa: E501
        if hasattr(self.infer_cfg, 'ice_template'):
            ice_template = ICL_PROMPT_TEMPLATES.build(
                self.infer_cfg['ice_template'])
            
        if self.concat_dataset:
            ice_templates = []
            for tmp_infer_cfg in self.infer_cfgs:
                ice_templates.append(ICL_PROMPT_TEMPLATES.build(tmp_infer_cfg['ice_template']))
            accum_datasets_len = []
            for l in self.per_dataset_len:
                accum_datasets_len.append(l)
                if len(accum_datasets_len) > 1:
                    accum_datasets_len[-1] += accum_datasets_len[-2]
        else:
            ice_templates = None
            accum_datasets_len = None

        if hasattr(self.infer_cfg, 'prompt_template'):
            prompt_template = ICL_PROMPT_TEMPLATES.build(
                self.infer_cfg['prompt_template'])

        retriever_cfg = self.infer_cfg['retriever'].copy()
        retriever_cfg['dataset'] = self.dataset
        retriever = ICL_RETRIEVERS.build(retriever_cfg)

        # set inferencer's default value according to model's config'
        inferencer_cfg = self.infer_cfg['inferencer']
        inferencer_cfg['model'] = self.model
        self._set_default_value(inferencer_cfg, 'max_out_len',
                                self.max_out_len)
        self._set_default_value(inferencer_cfg, 'min_out_len',
                                self.min_out_len)
        self._set_default_value(inferencer_cfg, 'batch_size', self.batch_size)
        inferencer_cfg['max_seq_len'] = self.model_cfg.get('max_seq_len')
        inferencer = ICL_INFERENCERS.build(inferencer_cfg)

        out_path = get_infer_output_path(
            self.model_cfg, self.dataset_cfg,
            osp.join(self.work_dir, 'predictions'))
        out_dir, out_file = osp.split(out_path)
        mkdir_or_exist(out_dir)

        if hasattr(self.infer_cfg, 'prompt_template') and \
                hasattr(self.infer_cfg, 'ice_template'):
            inferencer.inference(retriever,
                                 ice_template=ice_template,
                                 ice_templates=ice_templates,
                                 accum_datasets_len=accum_datasets_len,
                                 prompt_template=prompt_template,
                                 output_json_filepath=out_dir,
                                 output_json_filename=out_file,
                                 dataset_split_type=inferencer_cfg.get("dataset_split_type", "random"),
                                 random_idx=inferencer_cfg.get("random_idx", None),
                                 start_ratio=inferencer_cfg.get("start_ratio", 0.0),
                                 end_ratio=inferencer_cfg.get("end_ratio", 1.0))
        elif hasattr(self.infer_cfg, 'prompt_template'):
            inferencer.inference(retriever,
                                 ice_templates=ice_templates,
                                 accum_datasets_len=accum_datasets_len,
                                 prompt_template=prompt_template,
                                 output_json_filepath=out_dir,
                                 output_json_filename=out_file,
                                 dataset_split_type=inferencer_cfg.get("dataset_split_type", "random"),
                                 random_idx=inferencer_cfg.get("random_idx", None),
                                 start_ratio=inferencer_cfg.get("start_ratio", 0.0),
                                 end_ratio=inferencer_cfg.get("end_ratio", 1.0))
        else:
            inferencer.inference(retriever,
                                 ice_template=ice_template,
                                 ice_templates=ice_templates,
                                 accum_datasets_len=accum_datasets_len,
                                 output_json_filepath=out_dir,
                                 output_json_filename=out_file,
                                 dataset_split_type=inferencer_cfg.get("dataset_split_type", "random"),
                                 random_idx=inferencer_cfg.get("random_idx", None),
                                 start_ratio=inferencer_cfg.get("start_ratio", 0.0),
                                 end_ratio=inferencer_cfg.get("end_ratio", 1.0))

    def _set_default_value(self, cfg: ConfigDict, key: str, value: Any):
        if key not in cfg:
            cfg[key] = value
            
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


def parse_args():
    parser = argparse.ArgumentParser(description='Model Inferencer')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('--coeff_path', default=None, help='Path to the searched combination coefficients')
    parser.add_argument('--model_path', help='path to Mixtral-8x7B-Instruct-v0.1 huggingface model', required=True)
    parser.add_argument('--work_dir', type=str, default=None, help='Path to save generated results and metrics')
    args = parser.parse_args()
    return args

def save_routing_weights(model, path):
    os.makedirs(path, exist_ok=True)
    for i in range(len(model.model.model.layers)):
        torch.save(model.model.model.layers[i].block_sparse_moe.routing_weights_save, os.path.join(path, f"{i}-layer.pth"))
        
def load_coeff(coeff, model, expert_weights_dict, model_state_dict, budget=4, prefix="block_sparse_moe.experts"):
    merge_weights(model, coeff, expert_weights_dict, model_state_dict, target_num_expert=budget, prefix=prefix)

if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg["models"][0]["path"] = args.model_path
    cfg["models"][0]["tokenizer_path"] = args.model_path
    cfg["work_dir"] = args.work_dir
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
    
    if args.coeff_path is not None:
        print(f"Loading weight merging and router mapping matrix from {args.coeff_path}")
        coeff = torch.load(args.coeff_path)
        budget = list(coeff.values())[0].shape[0]
        load_coeff(coeff, model, expert_weights_dict, model_state_dict, budget=budget, prefix=prefix)
    
    cfg_save = copy.deepcopy(cfg)
    start_time = time.time()
    inferencer = OpenICLInferTask(cfg)
    evaler = OpenICLEvalTask(cfg_save)
    inferencer.run(model=model)
    end_time = time.time()
    get_logger().info(f'time elapsed: {end_time - start_time:.2f}s')
    evaler.run()

    
    
    
