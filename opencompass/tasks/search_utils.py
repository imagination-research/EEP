"""
A script used to define all utility functions for the search process.
"""
import os
import random
import torch
import numpy as np
import pathlib
import copy

# from transformers.models.qwen2_moe import modeling_qwen2_moe

def pattern_in(text, pattern):
    patterns = pattern.split(".")
    texts = text.split(".")
    for i in range(len(texts)):
        for j in range(len(patterns)):
            if patterns[j] == "*":
                continue
            elif "[" in patterns[j] and "]" in patterns[j]:
                tmp_pattern = patterns[j][1:-1].split("-")
                int_range = list(range(int(tmp_pattern[0]), int(tmp_pattern[1]) + 1))
                str_range = [str(x) for x in int_range]
                if texts[i + j] in str_range:
                    continue
                else:
                    break
            else:
                if texts[i + j] == patterns[j]:
                    continue
                else:
                    break
        else:
            return True
    return False

def get_cur_id(key, prefix):
    name_list = key.split(".")
    for i in range(len(name_list)):
        if name_list[i] == prefix:
            return int(name_list[i + 1])
        
def replace_cur_id(key, cur_id, new_id, prefix):
    name_list = key.split(".")
    new_key = ""
    for i in range(len(name_list)):
        if name_list[i - 1] == prefix:
            assert name_list[i] == str(cur_id)
            new_key += str(new_id)
        else:
            new_key += name_list[i]
        if i!=len(name_list)-1:
            new_key += "."
    return new_key

def list_sub(full, sub):
    result = []
    for i in range(len(full)):
        if not full[i] in sub:
            result.append(full[i])
    return result

def list_equal(list1, list2):
    for key in list1:
        if not key in list2:
            return False
    for key in list2:
        if not key in list1:
            return False
    return True

def shuffle_same(sorted_list, mode="train"):
    value_dict = {}
    for item in sorted_list:
        if mode == "train":
            if item[-1][1] not in value_dict.keys():
                value_dict[item[-1][1]] = [item]
            else:
                value_dict[item[-1][1]].append(item)
        elif mode == "val":
            if item[-1][0][1] not in value_dict.keys():
                value_dict[item[-1][0][1]] = [item]
            else:
                value_dict[item[-1][0][1]].append(item)            
    for key in value_dict.keys():
        random.shuffle(value_dict[key])

    result_list = [item for sublist in value_dict.values() for item in sublist]
    
    return result_list

def coeff_equal(coeff_1, coeff_2):
    if not list_equal(list(coeff_1.keys()), list(coeff_2.keys())):
        return False
    for group in coeff_1.keys():
        if isinstance(coeff_1[group], np.ndarray) or isinstance(coeff_1[group], torch.Tensor):
            if not(coeff_1[group] == coeff_2[group]).all():
                return False
        elif isinstance(coeff_1[group], list):
            if not list_equal(coeff_1[group], coeff_2[group]):
                return False
        else:
            import ipdb; ipdb.set_trace()
            
    return True

def get_expert_eliminate_init(num_expert=8, eliminate_expert=1):
    keep_experts = sorted(random.sample(range(num_expert), num_expert - eliminate_expert))
    full_matrix = torch.zeros(num_expert, num_expert)
    for i in range(num_expert):
        full_matrix[i, i] = 1.0
    group_matrix = full_matrix[keep_experts]
    group_coeff = group_matrix
    
    return group_coeff
    
def get_init(cfg, data_path, num_expert, all_init_path=None):
    coeff = {}
    eliminate_expert = num_expert - cfg["budget"]
    for key in cfg["weight_group"]:
        if not "scaling" in key and not "transfer" in key:
            group_coeff = get_expert_eliminate_init(num_expert=num_expert, eliminate_expert=eliminate_expert)
            coeff[key] = group_coeff
            trans_key = key.replace("layers", "transfer") if not "others" in key else "others_transfer"
            coeff[trans_key] = copy.deepcopy(group_coeff)
        elif "scaling" in key:
            coeff[key] = torch.ones([num_expert])

    init_data_num = 0
    for i in range(len(all_init_path)):
        init_data_num += 1
        if coeff_equal(torch.load(all_init_path[i])[0], coeff):
            return "continue enumeration", "continue enumeration"
    if init_data_num > cfg["max_init_num"]:
        print(f"The number of evaluated inits has already surpassed the max number")
        return None, None
    try:
        torch.save([coeff], os.path.join(data_path, "init", "init" + str(init_data_num) + "_place_holder"))
    except:
        # import ipdb; ipdb.set_trace()
        return "continue enumeration", "continue enumeration"
    return coeff, os.path.join(data_path, "init", "init" + str(init_data_num) + "_place_holder")

    return None, None

def get_all_data(data_path, metric_cfg):
    data_paths = [file for file in pathlib.Path(data_path).glob('**/*.{}'.format("pth"))]
    get = False
    if metric_cfg.get("switch_to_dev", False):
        p = random.uniform(0, 1)
        use_train_prob = metric_cfg.get("use_train_prob", 0.5)
    while not get:
        try:
            print(f"Getting all evaluated weights...")
            population = []
            for path in data_paths:
                data = torch.load(path)
                if metric_cfg.get("switch_to_dev", False):
                    if p > use_train_prob and len(data[2]) > 1:
                        population.append((data[0], data[2][0]))
                    else:
                        population.append(data[:2])
                else:
                    population.append(data[:2])
            get = True
        except:
            print(f"Failed to get all evaluated scores, wait...")
            import ipdb; ipdb.set_trace()
            get = False  

    if metric_cfg["better"] == "larger":
        sort_key = lambda x: -x[-1][1]
    elif metric_cfg["better"] == "smaller":
        sort_key = lambda x: x[-1][1]
    else:
        raise NotImplementedError
    population = sorted(population, key=sort_key)
    population = shuffle_same(population)
    return population

def get_all_discrete_data_with_val_acc(data_path, metric_cfg):
    data_paths = [file for file in pathlib.Path(data_path).glob('**/*.{}'.format("pth"))]
    get = False
    while not get:
        try:
            print(f"Getting all evaluated weights...")
            population = []
            for path in data_paths:
                data = torch.load(path)
                if check_discrete(data[0]):
                    population.append(data)
            get = True
        except:
            print(f"Fail to get all evaluated scores, wait and restart this process...")
            import ipdb; ipdb.set_trace()
            get = False  

    if metric_cfg["better"] == "larger":
        sort_key = lambda x: -x[-1][0][1]
    elif metric_cfg["better"] == "smaller":
        sort_key = lambda x: x[-1][0][1]
    else:
        raise NotImplementedError
    population = sorted(population, key=sort_key)
    population = shuffle_same(population, mode="val")
    return population

def crossover(cfg, population, group_change):
    print("Begin crossover operation")
    if cfg["crossover"]["parents_select_type"] == "ranking":
        parents_1, parents_2 = random.sample(population[:cfg["crossover"]["parents_rank"]], 2)
    elif cfg["crossover"]["parents_select_type"] == "absolute":
        best = population[0][1][1]
        threshold = cfg["crossover"]["parents_threshold"]
        for i in range(1, len(population)):
            if cfg["metric"][cfg["metric"]["type"]]["better"] == "larger":
                if best - population[i][1][1] > threshold and best - population[i - 1][1][1] <= threshold:
                    break
            else:
                raise NotImplementedError
        i = max(i, cfg["crossover"]["parents_rank"])
        parents_1, parents_2 = random.sample(population[:i], 2)
    else:
        raise NotImplementedError
    
    try:
        assert list_equal(list(parents_1[0].keys()), list(parents_2[0].keys()))
        assert list_equal(list(parents_1[0].keys()), cfg["weight_group"])
    except:
        import ipdb; ipdb.set_trace()
    new_coeff = {}
    for key in cfg["weight_group"]:
        if group_change[key] == 1:
            num_expert = parents_1[0][key].shape[1]
            target_expert = parents_1[0][key].shape[0]
            new_matrix = torch.zeros(target_expert, num_expert)
            for i in range(target_expert):
                if random.uniform(0, 1) < cfg["crossover"]["from_front_prob"]:
                    new_matrix[i] = parents_1[0][key][i]
                else:
                    new_matrix[i] = parents_2[0][key][i]
            new_coeff[key] = new_matrix
        else:
            if random.uniform(0, 1) < cfg["crossover"]["from_front_prob"]:
                new_coeff[key] = parents_1[0][key]
            else:
                new_coeff[key] = parents_2[0][key]
                    
    return new_coeff

def discrete_mutate(cfg, coeff, group_change):
    print(f"Begin discrete mutation. Parents: {coeff}")
    
    new_coeff = copy.deepcopy(coeff)
    check_discrete(coeff)
    for key in new_coeff.keys():
        if not "transfer" in key:
            if group_change[key] == 1:
                target_expert, num_expert = coeff[key].shape[0], coeff[key].shape[1]
                trans_key = key.replace("layers", "transfer") if not "others" in key else "others_transfer"
                for i in range(target_expert):
                    if random.uniform(0, 1) < cfg["discrete_mutate"]["prob"]:
                        # print("----------")
                        # print(f"change key: {key} | line: {i}")
                        # print(f"Before change: {new_coeff[key][i]}")
                        for j in range(num_expert):
                            new_coeff[key][i, j] = 0.0
                            new_coeff[trans_key][i, j] = 0.0
                        while(1):
                            new_ind = random.randint(0, num_expert - 1)
                            for k in range(target_expert):
                                if new_coeff[key][k, new_ind] == 1.0:
                                    break
                            else:
                                break
                        new_coeff[key][i, new_ind] = 1.0
                        new_coeff[trans_key][i, new_ind] = 1.0
                        # print(f"After change: {new_coeff[key][i]}")
    check_discrete(new_coeff)
    return new_coeff

def mutate(cfg, coeff, group_change):
    # print(f"Begin mutation. Parents: {coeff}")
    new_coeff = copy.deepcopy(coeff)
    for key in new_coeff.keys():
        if group_change[key] == 1:
            target_expert, num_expert = coeff[key].shape[0], coeff[key].shape[1]
            for i in range(target_expert):
                for j in range(num_expert):
                    if random.uniform(0, 1) < cfg["mutate"]["prob"]:
                        new_coeff[key][i, j] = new_coeff[key][i, j] + random.gauss(0, 1) * cfg["mutate"]["scale"]
                
    return new_coeff

def get_metric(results, cfg):
    assert len(results) == 1, "Currently only evaluating one model is supported"
    if cfg["metric"]["type"] in ["accuracy", "score", "old_score"]:
        metric = results[0][0][cfg["metric"]["type"]]
        return (cfg["metric"]["type"], metric)
    elif cfg["metric"]["type"] == "rouge1+rouge2":
        metric = results[0][0]['rouge1'] + results[0][0]['rouge2']
        return (cfg["metric"]["type"], metric)
    else:
        raise NotImplementedError
    
def check_discrete(coeff):
    for key in coeff.keys():
        target_expert, num_expert = coeff[key].shape[0], coeff[key].shape[1]
        for i in range(target_expert):
            total = 0
            for j in range(num_expert):
                total += coeff[key][i, j]
                try:
                    assert coeff[key][i, j] == 1.0 or coeff[key][i, j] == 0.0
                except:
                    import ipdb; ipdb.set_trace()
            try:
                assert total == 1.0
            except:
                import ipdb; ipdb.set_trace()
        
        if not "transfer" in key and not "scaling" in key:
            trans_key = key.replace("layers", "transfer") if not "others" in key else "others_transfer"
            try:
                assert (coeff[trans_key] == coeff[key]).all()
            except:
                import ipdb; ipdb.set_trace()
    
def check(cfg, coeff):
    if coeff is None:
        return

    target_expert = None
    for key in coeff.keys():
        if not "scaling" in key:
            try:
                assert len(coeff[key].shape) == 2
            except:
                import ipdb; ipdb.set_trace()
            if target_expert is None:
                target_expert = coeff[key].shape[0]
            else:
                try:
                    assert coeff[key].shape[0] == target_expert
                except:
                    import ipdb; ipdb.set_trace()
        else:
            try:
                assert isinstance(coeff[key], torch.Tensor)
                assert len(coeff[key].shape) == 1
            except:
                print(coeff[key])
                import ipdb; ipdb.set_trace()    