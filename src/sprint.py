"""
File: sprint.py
- A main file for SPRINT (Sublayer PRuning wIth LateNcy and Tunability Information)
- This source code is written based on the following GitHub repository:
    https://github.com/jiwonsong-dev/SLEB
"""

import fire
import time
import torch
import os
from collections import OrderedDict

from glob import glob
from pathlib import Path
from torch import nn
from tqdm import tqdm

# from utils.model_utils import get_llm
from utils.onoff_llama import convert, turn_off
from utils.data_utils import *
from utils.eval_utils import comprehensive_eval
from utils.general_utils import *
from utils.compression_utils import *
from utils.CompressionManager import *
from transformers import set_seed

def sprint(
    # Arguments for initializing
    model_name:                 str     = 'meta-llama/Meta-Llama-3-8B',
    dataset:                    str     = 'wikitext2',
    nsamples:                   int     = 128,
    seed:                       int     = 0,
    num_cpu_layers:             int     = 0,
    num_remove_sublayers:       int     = -1,
    target_speedup:             float   = 1.61,
    output_dir:                 str     = '../outputs/',
    cache_dir:                  str     = '../cache',
    result_file:                str     = '_results.csv',
    logfile:                    str     = "log",
    pruned_model_file:          str     = 'pruned_model.pickle',
    
    # Arguments for intermediate evaluation
    eval_every_step:            bool    = False,
    eval_steps:                 str     = '',
    eval_speedups:              str     = '',

    # Arguments for speedup estimation
    generation:                 bool    = True,
    generation_length:          int     = 512,
    prompt_length:              int     = 1024,
    speedup_batch_size:         int     = 1,
    speedup_iteration:          int     = 3,

    # Arguments for importance scoring
    # General
    sensi_batch_size:           int     =   8,
    metrics:                    str     =   "l2",
    
    # (I1) Latency-aware importance scoring
    latency_aware:              bool    =   True,

    # (I2) Tunability-aware sensitivity evaluation
    tuning_sublayer:        str     =   'mlp',
    in_comp_tuning:             bool    =   True,
    update_tuned_weights:       bool    =   True,
    in_comp_tuning_dtype:       str     =   'float',
    direction:                  str     =   'out_channel',
    in_comp_tuning_ratio:       float   =   1.0,
    damping_coefficient:        float   =   0.,

    # (I3) Avoiding unnecessary computations
    num_checkpoints:            int     =   8,
    checkpointing_on_cpu:       bool    =   False,
    num_candidate_sublayers:    int     =   5,
    
):
    """
    Main function for running SPRINT
    For a detailed explanation of the arguments, please refer to the ReadMe.md file.
    """

    # Initialiation
    # Generate folders and logfile
    set_seed(seed) # set random seed for reproducibility
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    log_dir = Path(output_dir)
    logger = create_logger(log_dir=log_dir, log_filename=logfile)
    result_file_path = os.path.join(output_dir, result_file)
    pruned_model_path = os.path.join(output_dir, pruned_model_file)

    # Load latency dict
    latency_dict = get_latencies(model_name,
                                 generation         =  generation,
                                 generation_length  =  generation_length,
                                 prompt_length      =  prompt_length,
                                 batch_size         =  speedup_batch_size,
                                 iteration          =  speedup_iteration,
                                 cache_dir          =  cache_dir)
    
    clear_gpu_memory()

    # Load model
    model = get_llm(model_name, 'cpu')
    use_cache = model.config.use_cache
    model.config.use_cache = False
    model_type = model.name.split("/")[-1]
    model = convert(model) # for pruning sublayers
    model.eval()
    num_sublayers = len(model.model.layers) * 2

    # Load dataset 
    cache_file = f'{cache_dir}/{model_type}_{dataset}_{nsamples}_{seed}.pkl'
    cache_file_test = f'{cache_dir}/{model_type}_{dataset}_test.pkl'

    print(f"* Load {dataset} dataset ->  ", end='')
    if os.path.isfile(cache_file):
        print("find the cached dataloader -> ", end='')
        dataloader = torch.load(cache_file)
    else:
        print("Generate and cache dataloader -> ", end='')
        dataloader = get_loaders(dataset,
                                nsamples=nsamples,
                                seed=seed,
                                model=model_name,
                                batch_size=1
                                )[0]
        torch.save(dataloader, cache_file)
    if os.path.isfile(cache_file_test):
        testenc = torch.load(cache_file_test)
    else:
        testenc = get_loaders(dataset,
                        nsamples=nsamples,
                        seed=seed,
                        model=model_name,
                        batch_size=1
                        )[1]
        torch.save(testenc, cache_file_test)
    print("Done")

    # Preprocess arguments
    
    # Set metrics for measuring sensitivities
    # You can use l1 norm, l2 norm, and cos dissimilarity
    if type(metrics) == str:
        metrics_list = [metric.strip() for metric in metrics.split(',')]  
    else:
        # tuple
        metrics_list = list(metrics)

    # Set data type for in-compression tuning
    if 'float' in in_comp_tuning_dtype or 'fp32' in in_comp_tuning_dtype:
        _in_comp_dtype = torch.float32
    else:
        _in_comp_dtype = torch.float64

    # Set evaluation conditions
    if type(eval_steps) == tuple:
        eval_steps = list(eval_steps)
    elif type(eval_steps) == int:
        eval_steps = [int(eval_steps)]
    else:
        eval_steps = []

    if type(eval_speedups) == tuple:
        eval_speedups = list(eval_speedups)
    elif type(eval_speedups) == float:
        eval_speedups = [float(eval_speedups)]
    else:
        eval_speedups = []

    # Device settings for logging
    attn_type = str(type(model.model.layers[0].self_attn)).split('.')[-1][:-2]
    gpu_num = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())  

    # Logging experimental settings
    setting_str  = "\n" + "==" * 45
    setting_str += f"\n[Experimental Settings]"
    setting_str += "\n* General"
    setting_str += f"\n - Model: {model.name} | sample data: {nsamples} {dataset} | seed: {seed}"
    setting_str += f"\n - Numer of sublayers to remove: {num_remove_sublayers} | target inference speedup: {target_speedup:.2f}"
    setting_str += f"\n - Evaluation steps: {eval_steps} | evaluation speedups: {eval_speedups}"
    
    setting_str += "\n\n* Latency estimation"
    setting_str += f"\n - Attention type: {attn_type} | GPU: {gpu_num} X {gpu_name}" 
    setting_str += f"\n - Genernation length: {generation_length} | " + \
                            f"prompt length: {prompt_length} | " + \
                            f"batch size: {speedup_batch_size} | " + \
                            f"iteration {speedup_iteration}"
    setting_str += f"\n - Estimated latencies: " + \
                   f"MHA ({latency_dict['MHA']:.2f}) | " + \
                   f"MLP ({latency_dict['MLP']:.2f})"
    
    setting_str += "\n\n* Importance scoring"
    setting_str += f"\n - Sensitivity metrics:  {metrics_list} | batch size: {sensi_batch_size} "
    setting_str += f"\n - Number of layers on CPU: {num_cpu_layers} | number of checkpoints: {num_checkpoints}"
    setting_str += f"\n - latency aware: {latency_aware} | tuning sublayer type: {tuning_sublayer}"

    setting_str += f"\n - In-compression tuning: {in_comp_tuning} ({direction}, ratio: {in_comp_tuning_ratio}, update: {update_tuned_weights})"
    setting_str += f"\n - Number of Candidates: {num_candidate_sublayers}"

    setting_str += f"\n - Number of checkpoints: {num_checkpoints} (CPU: {checkpointing_on_cpu})"
    setting_str += "\n" + "==" * 45 + "\n"

    logger.info(setting_str)

    # Initialize compression managers
    comp_manager = CompressionManager(num_sublayers, 
                                      nsamples, 
                                      metrics_list,
                                      latency_dict,
                                      latency_aware)
    if in_comp_tuning:
        in_comp_manager = CompressionManager(num_sublayers, 
                                             nsamples, 
                                             metrics_list,
                                             latency_dict,
                                             latency_aware)
        # Set all scores as infinite for intialization
        in_comp_manager.set_score_inf()
        tuned_weights = {}
    
    # Dictionary for saving pruned sublayer indices and tuned weights
    params_to_save = OrderedDict()


    # Get device mapping dictionary
    device_map, chcekpoint_indices = get_device_map(model.model.layers, 
                                                    num_checkpoints=num_checkpoints,
                                                    num_cpu_layers=num_cpu_layers,
                                                    logger=logger)
    st_device = 'cuda:0'

    # Get inputs
    inps, attention_mask_batch, position_ids = \
        get_inputs(model, dataloader, nsamples, sensi_batch_size, st_device)
    
    # Initialize variables
    checkpoints = {0: inps.cpu()}
    cand_time = 0
    in_comp_time = 0
    other_time = 0
    start_subid = 0
    _count = 0
    target_subids = []
    speedup = comp_manager.get_speedup()

    # Timing
    start_point = time.time()
    while True:
        # Termination condition
        if target_speedup >= 1:
            if speedup >= target_speedup:
                break
        elif num_remove_sublayers >= 1:
            if _count >= num_remove_sublayers:
                break
        else:
            raise Exception("You must set either target_speedup or num_remove_sublayers")
        
        logger.info('\n\n')
        _count += 1
        
        if len(target_subids) == 0:
            # Importance scoring
            time_start_cand = time.time()
            # Stage 1: candidate selection
            candidate_selection(
                model                 = model, 
                start_subid           = start_subid, 
                chcekpoint_indices    = chcekpoint_indices, 
                checkpoints           = checkpoints, 
                device_map            = device_map,
                checkpointing_on_cpu  = checkpointing_on_cpu,                                   
                attention_mask_batch  = attention_mask_batch, 
                position_ids          = position_ids, 
                comp_manager          = comp_manager,
                batch_size            = sensi_batch_size, 
                tuning_sublayer   = tuning_sublayer
                )
            
            cand_time += time.time() - time_start_cand

            # For updating start_subid
            if num_checkpoints > 0:
                start_subid = 9999

            if in_comp_tuning:
                # Importance scoring before in-compression tuning
                time_start_in_comp = time.time()
                candidate_subids = comp_manager.get_target_subids(num_candidate_sublayers)

                # Finding cached weights to avoid recomputation
                _id = len(candidate_subids)-1
                while _id >= 0:
                    _sid = candidate_subids[_id]
                    if _sid in tuned_weights.keys(): # already tuned
                        del candidate_subids[_id]
                    _id -=1
                logger.info(f" * Cached weights: {tuned_weights.keys()}")
                logger.info(f" * Filtered candidate subids: {candidate_subids}")

                # Stage2: importance scoring with in-compression tuning
                tuned_weights = importance_scoring(
                    model                   = model, 
                    candidate_subids        = candidate_subids, 
                    chcekpoint_indices      = chcekpoint_indices, 
                    checkpoints             = checkpoints, 
                    device_map              = device_map,
                    tuned_weights           = tuned_weights, 
                    attention_mask_batch    = attention_mask_batch, 
                    position_ids            = position_ids,
                    in_comp_manager         = in_comp_manager, 
                    batch_size              = sensi_batch_size,  
                    tuning_sublayer         = tuning_sublayer, 
                    in_comp_dtype           = _in_comp_dtype,
                    in_comp_tuning_ratio    = in_comp_tuning_ratio, 
                    direction               = direction,
                    damping_coefficient     = damping_coefficient
                )
                target_subids = in_comp_manager.get_target_subids(1)

                in_comp_time += time.time() - time_start_in_comp
            else:
                target_subids = comp_manager.get_target_subids(1)
                logger.info(f"Compression target sublayers: {target_subids}")
            
        start_time_others = time.time()
        # Select the pruning target and prune it
        target_subid = target_subids.pop(0)
        turn_off(model, target_subid)
        comp_manager.update_status(target_subid, status=False)
        if in_comp_tuning:
            in_comp_manager.update_status(target_subid, status=False)
        speedup = comp_manager.get_speedup()

        if in_comp_tuning:
            # Update tuned weights
            with torch.no_grad():
                tuned_sid, _tuned_weights = tuned_weights.pop(target_subid)
                if update_tuned_weights and _tuned_weights is not None:
                    load_tuned_sublayer(model.model.layers, tuned_sid, _tuned_weights)

            # Remove saved tuned_weights whose layer index is higher than the pruned one
            for _id in list(tuned_weights.keys()):
                _tuned_id = tuned_weights[_id][0]
                if _tuned_id >= target_subid:
                    del tuned_weights[_id]
                    in_comp_manager.set_score_inf(_id)
            # update 

        # Find the loweset sublayer to update for the next iteration
        if num_checkpoints > 0:
            i = num_sublayers-1
            while comp_manager.tuning_indices[i] >= target_subid:
                i-=1
            update_start = i
            j = -1
            while chcekpoint_indices[j] > update_start:
                j -= 1
            cached_start_subid = chcekpoint_indices[j]
            start_subid = min(start_subid, cached_start_subid)
        else:
            cached_start_subid = 0
        
        # Evaluation if we need
        other_time += time.time()-start_time_others
        step_str  = f'\n[Step {_count}] Prune {target_subid:3d}th sublayer | Speedup: {speedup:.4f}' 
        step_str += f"\n - Pruned sublayers: {comp_manager.get_pruned_sublayer_list()}"
        step_str += f"\n - Running time: {cand_time + in_comp_time + other_time:.2f}s " + \
                    f"[cand./ tuning/ others]: [{cand_time:.2f}s / {in_comp_time:.2f}s/ {other_time:.2f}s] "        
        logger.info(step_str)
        
        # Evaluation 
        do_eval = ((eval_every_step) or \
                  (_count in eval_steps) or \
                  (len(eval_speedups) > 0 and eval_speedups[0] < speedup))
        if do_eval:
            if (num_cpu_layers !=0 or  torch.cuda.device_count()!=1):
                print("You cannot do intermediate evaluation if you set cpu layers or multi gpus")
                do_eval = False

        if len(eval_speedups) > 0 and eval_speedups[0] < speedup:
            print(f"Eval speepup: {eval_speedups} -> ", end='')
            eval_speedups.pop(0)
            print(f"{eval_speedups}")

        if do_eval:
            logger.info(f"\n\n[Step:{_count:2d}] Pruned sublayers: " +  
                        f"{comp_manager.get_pruned_sublayer_list()}")
            tmp_str = comprehensive_eval(model)
            # # logger.info(f"After tuning: {tmp_str}")
            with open(result_file_path, 'a') as file:
                file.write(f'{_count},{speedup:.2f},{target_subid},{tmp_str}\n')
        
        # Save the pruned model
        if in_comp_tuning:
            for _k in _tuned_weights.keys():
                if len(_tuned_weights[_k]) == 2:
                    _tuned_channels, _mask = _tuned_weights[_k]
                    _tuned_weights[_k] = (_tuned_channels.cpu(), _mask.cpu())
                else:
                    _tuned_channels, _mask, _direction = _tuned_weights[_k]
                    _tuned_weights[_k] = (_tuned_channels.cpu(), 
                                          _mask.cpu(),
                                          _direction)
            params_to_save[target_subid] = (tuned_sid, _tuned_weights)
        else:
            params_to_save[target_subid] = (None, None)
        torch.save(params_to_save, pruned_model_path)

    print(f"* Running time: {time.time() - start_point:10.3f} sec (including evaluation)")
    model.config.use_cache = use_cache

if __name__ == "__main__":
    fire.Fire(sprint)
