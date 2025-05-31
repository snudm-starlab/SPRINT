"""
File: general_utils.py
- General utility functions for running SPRINT
"""

import gc
import logging
import os, sys
import torch
from math import inf
from termcolor import colored
from torch import nn
from typing import Union, Tuple, Iterator
from transformers import AutoModelForCausalLM
import psutil
import copy


# [Section 1] activation-related functions
@torch.no_grad()
def get_inputs(
    model,
    dataloader: list,
    nsamples: int,
    batch_size: int,
    device: Union[str, torch.device]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get inputs of the very first layer of a given model

    Args:
        model (_type_): the target model to prune
        dataloader (list): dataloader
        nsamples (int): the number of data
        batch_size (int): batch size
        device (Union[str, torch.device]): The device on which the inputs are acquired

    Returns:
        inps (torch.Tensor): an acquired inputs
        attention_mask_batch (torch.Tensor): a batch of attention masks
        position_ids (torch.Tensor): position ids
    """
    
    
    inps = torch.zeros(
        (nsamples, model.seqlen, model.model.config.hidden_size)
    ).half().to(device)

    if device == "cpu" or device == torch.device("cpu"):
        _device = "cuda"
    else:
        _device = device
    
    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(_device)
    model.model.norm = model.model.norm.to(_device)
    cache = {"i": 0}

    layers[0] = layers[0].to(_device)

    # Catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0]).to(_device)

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= inps.shape[0]:
                break
            try:
                model(batch[0].to(_device))
            except ValueError:
                pass
    
    layers[0] = layers[0].module
    layers[0] = layers[0].to(device)
    # We do not use embed tokens and norm any more. Move them to CPU
    model.model.embed_tokens = model.model.embed_tokens.to('cpu')
    model.model.norm = model.model.norm.to('cpu')

    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(batch_size,1,1,1)
    else:
        attention_mask_batch = None

    clear_gpu_memory()
    return inps, attention_mask_batch, position_ids

class StopForwardException(Exception):
    """
    An exception class for hijacking activations
    """
    pass


def hijack(module, _list, _hijack_input, _stop_forward=False, dtype=torch.float16):
    """
    hijack the input or outuput of the module and append it to the _list
    if _stop_forward=True, then it raise an error after forwarding the module

    Args:
        module: the target module for hijacking activations
        _list: a list to accumulate hijacked activations
        _hijack_input (bool): hijacking inputs or outputs (True: inputs, False: outputs)
        _stop_forward (bool, optional): stop forwarding after hijacking. Defaults to False.
        dtype (torch.dtype): data type. Defaults to torch.float16.

    Returns:
       handle: a handle to remove the generated hook function
    """
    if _hijack_input:
        def input_hook(_, inputs):
            _list.append(inputs[0].clone().data.to(dtype=dtype))
            if _stop_forward:
                raise StopForwardException

        handle = module.register_forward_pre_hook(input_hook)
    else:
        def output_hook(_, __, outputs):
            if isinstance(outputs, tuple):
                _new_list = []
                for _output in outputs:
                    if isinstance(_output, torch.Tensor):
                        _new_list.append(_output.clone().data.to(dtype=dtype))
                    else:
                        _new_list.append(_output)
                _list.append(tuple(_new_list))
            else:
                _list.append(outputs.clone().data.to(dtype=dtype))
            if _stop_forward:
                raise StopForwardException

        handle = module.register_forward_hook(output_hook)
    return handle 

def check_and_match_devices(layers, 
                            sid, 
                            tuning_sid, 
                            device_map, 
                            inps=None, 
                            attention_mask_batch=None, 
                            position_ids=None,
                            is_preprocess=True):
    """
    A utility function that aligns the devices of inputs and layers

    Args:
        layers (nn.Module): layers of the input model
        sid (int): the index of an input sublayer
        tuning_sid (int): the index of an sublayer to tune
        device_map (dict): dictionary for device mapping
        inps (torch.Tensor): _description_. Defaults to None.
        attention_mask_batch (torch.Tensor): attention masks
        position_ids (torch.Tensor, optional): position ids
        is_preprocess (bool, optional): whether its preprocess or not

    Returns:
        _type_: _description_
    """
    if is_preprocess:
        # Move layers to proper device
        for _i in range(sid//2, tuning_sid//2+1):
            if device_map[_i] == 'cpu':
                _device = 'cuda:0'
            else:
                _device = device_map[_i]
            layers[_i] = layers[_i].to(_device)
        # Move inputs to the first layer's device
        if inps is not None:
            inps, attention_mask_batch, position_ids = \
                move_inputs(inps, attention_mask_batch, position_ids, layers[sid//2])
    else:
        # Post-processing
        # Move layers to the proposer device
        for _i in range(sid//2, tuning_sid//2+1):
            _device = device_map[_i]
            layers[_i] = layers[_i].to(_device)
    return inps, attention_mask_batch, position_ids


def move_inputs(
    inps: torch.Tensor,
    atts: torch.Tensor,
    pos: torch.Tensor,
    layer,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Move the device of inputs
    """

    device = get_device(layer)
    if inps is not None and inps.device != device:
        inps = inps.to(device)
    if atts is not None and atts.device != device:
        atts = atts.to(device)
    if pos is not None and pos.device != device:
        pos = pos.to(device)    
    return inps, atts, pos

def batch_forward_multiple(
    layers: nn.ModuleList,
    start: int,
    final: int, 
    inps_batch: torch.Tensor,
    attention_mask_batch: torch.Tensor,
    position_ids: torch.Tensor,
    is_gt: bool,
    skip: bool = False,
    is_layer: bool = False,
) -> torch.Tensor:
    """
    Forward propagation from `subid` to `final' sublayers

    Args:
        layers (nn.ModuleList): layers
        start (int): the index of the lowest sublayer to forward
        final (int): the index of the highest sublayer to forward
        inps_batch (torch.Tensor): a batch of input activations
        attention_mask_batch (torch.Tensor): a batch of attention masks
        position_ids (torch.Tensor): position ids
        is_gt (bool): forwarding for an unpruned model or not.
        skip (bool, optional): skipping the lowest sublayer or not. Defaults to False.
        is_layer (bool, optional): layerwise pruning or not. Defaults to False.

    Returns:
        inps_batch (torch.Tensor): the forwarded activations
    """
    
    if is_layer:    
        # CASE1: layer
        with torch.cuda.amp.autocast():
            for idx in range(start, final + 1):
                layer = layers[idx]

                if is_gt:
                    orig_status_mha = layer.pass_mha
                    orig_status_mlp = layer.pass_mlp
                    layer.pass_mha = False; layer.pass_mlp = False
                inps_batch = layer(inps_batch, attention_mask_batch, position_ids)[0]
                if is_gt:
                    layer.pass_mha = orig_status_mha
                    layer.pass_mlp = orig_status_mlp
    
    else:
        # CASE2: sublayer  
        # skip the target sublayer for simulating pruned status
        st_subid = start + 1 if skip else start
        with torch.cuda.amp.autocast():
            for sid in range(st_subid, final + 1):
                layer = layers[sid // 2]
                is_mlp = (sid % 2 == 1)
                if inps_batch.device != get_device(layer):
                    inps_batch = inps_batch.to(device=get_device(layer))
                if is_mlp:
                    if is_gt:
                        orig_status = layer.pass_mlp
                        layer.pass_mlp = False
                    inps_batch = layer.do_mlp_forward(inps_batch)
                    if is_gt:
                        layer.pass_mlp = orig_status
                else:
                    if is_gt:
                        orig_status = layer.pass_mha
                        layer.pass_mha = False
                    
                    inps_batch = layer.do_mha_forward(inps_batch, 
                                                    attention_mask_batch, 
                                                    position_ids)[0] 
                    if is_gt:
                        layer.pass_mha = orig_status
    return inps_batch

# [Section 2] model-related functions

def get_llm(model_name, device_map="auto"):
    """
    Return an LLM model
    """

    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 torch_dtype=torch.float16,
                                                 low_cpu_mem_usage=True,
                                                 device_map=device_map,
                                                 )
    model.seqlen = 2048
    model.name = model_name

    return model


def get_tuning_subid(
    subid:  int,
    tuning_sublayer: str,
    pruning_status: list,
) -> int:
    """_summary_

    Args:
        subid (int): the index of sublayer for importance measuring
        tuning_sublayer (str): the sublayer type for tuning
        pruning_status (list): the pruning status of sublayers

    Returns:
        int: the index of tuning sublayer
    """
    # Ablate tuning
    if tuning_sublayer is None:
        return subid
    final_subid = subid+1
    for sid in range(subid+1, len(pruning_status)):
        if pruning_status[sid] == False: continue
        if sid % 2 == 1 and tuning_sublayer.lower() == 'mlp':
            final_subid = sid
            break
        
        if sid %2 ==0 and tuning_sublayer.lower() == 'mha':
            final_subid = sid
            break

    if final_subid >= len(pruning_status):
        final_subid = len(pruning_status) - 1   
    return final_subid

def get_layernorm(layers, sid):
    """
    Return the layer normalization module corresponding to the given sublayer index
    """
    target_layer = layers[sid//2]
    if sid%2 == 1: # mlp
        return target_layer.post_attention_layernorm
    else: # mha
        return target_layer.input_layernorm
    
def get_output_proj(layers, subid):
    """
    Return the output projection module corresponding to the given sublayer index
    """
    _layer = subid//2; is_mha = (subid%2==0)
    if is_mha:
        return layers[_layer].self_attn.o_proj
    else:
        return layers[_layer].mlp.down_proj

@torch.no_grad()
def load_pruned_model(model, pruned_model_path, num_remove=-1, load_weight=True):
    """
    Load the pruned model suing the saved results

    Args:
        model (_type_): a target model to prune
        pruned_model_path: the path of the pruning result
        num_remove (int, optional): the number of sublayer to prune. Defaults to -1.
        load_weight (bool, optional): whether load the tuned weights or not. Defaults to True.

    Returns:
        model: the loaded pruned model
        pruned_sublayers: the list of pruned sublayer's indices
    """
    # (key = pruned sublayer id) : (value = (tuned_sublyer_id, tuned_weight))
    params_to_update = torch.load(pruned_model_path)
    # print(params_to_update)
    layers = model.model.layers
    pruned_sublayers = []
    
    assert len(params_to_update) >= num_remove, \
        f"Cannot remove {num_remove} sublayers more than the number of pruned " + \
        f"sublayers ({len(params_to_update)}) wirtten in the 'param_to_update'"

    # update params
    if not load_weight:
            print("Do not load tuned parameters")
    for count, pruned_sid in enumerate(params_to_update.keys()):
        if num_remove > 0 and count >= num_remove:
            break
        pruned_sublayers.append(pruned_sid)
        tuned_sid, _tuned_weights = params_to_update[pruned_sid]
        if tuned_sid is not None and load_weight:
                load_tuned_sublayer(layers, tuned_sid, _tuned_weights)
    model = sublayer_remove(model, kill_list=pruned_sublayers)
    return model, pruned_sublayers

def _update_weight(orig_weight, tuned_weights):
    """
    Update a weight using the tuning results
    Args:
        orig_weight (torch.Tensor): an original weight
        tuned_weights (dict): a dictionary of tuning results

    Returns:
        new_weight (torch.Tensor): a tuned weight
    """
    new_weight = copy.deepcopy(orig_weight)
    if len(tuned_weights) == 2 or tuned_weights[2] == 'out_channel':
        new_weight[tuned_weights[1], :] = tuned_weights[0].to(orig_weight.device)
    elif tuned_weights[2] == 'in_channel':
        new_weight[:, tuned_weights[1]] = tuned_weights[0].to(orig_weight.device)
    else:
        raise Exception("Uknwon channel direction")
    return new_weight

def load_tuned_sublayer(layers, sid, _tuned_weights):
    """
    Load a tuned weight of a sublayer

    Args:
        layers: layers
        sid (int): the index of sublayer to load
        _tuned_weights (torch.Tensor): a dictionary of tuned weights
    """
    layer = layers[sid//2]
    if sid % 2 == 0: # MHA
        if 'q_proj' in _tuned_weights:
            layer.self_attn.q_proj.weight.data = \
                _update_weight(layer.self_attn.q_proj.weight.data,
                            _tuned_weights['q_proj'])

        if 'k_proj' in _tuned_weights:
            layer.self_attn.k_proj.weight.data = \
                _update_weight(layer.self_attn.k_proj.weight.data,
                            _tuned_weights['k_proj'])

        if 'v_proj' in _tuned_weights:
            layer.self_attn.v_proj.weight.data = \
                _update_weight(layer.self_attn.v_proj.weight.data,
                            _tuned_weights['v_proj'])

        layer.self_attn.o_proj.weight.data = \
                _update_weight(layer.self_attn.o_proj.weight.data,
                            _tuned_weights['o_proj'])

    else: # MLP
        if 'up_proj' in _tuned_weights:
            layer.mlp.up_proj.weight.data = \
                _update_weight(layer.mlp.up_proj.weight.data,
                            _tuned_weights['up_proj'])
            
        if 'gate_proj' in _tuned_weights:
            layer.mlp.gate_proj.weight.data = \
                _update_weight(layer.mlp.gate_proj.weight.data,
                            _tuned_weights['gate_proj'])

        layer.mlp.down_proj.weight.data = \
                _update_weight(layer.mlp.down_proj.weight.data,
                            _tuned_weights['down_proj'])

def sublayer_remove(model, kill_list, _print=False):
    """
    removing sublayers of a langugage model

    Args:
        model: the target model to prune
        kill_list (_list): the list of sublayer indices to remove
        _print (bool, optional): whether print or not

    Returns:
        model: the pruned model
    """
    if _print:
        print(f"Removing sublayers... {kill_list}")
    return sublayer_remove_llama(model, kill_list, _print=_print)

def sublayer_remove_llama(model, kill_list, _print=False):
    """
    removing sublayers of llama model

    Args:
        model: the target model to prune
        kill_list (_list): the list of sublayer indices to remove
        _print (bool, optional): whether print or not

    Returns:
        model: the pruned model
    """
    _kill_list = copy.deepcopy(kill_list)
    _kill_list.sort() # sublayer indices
    layers = model.model.layers

    # Turn off sublayers to remove
    while(len(_kill_list)>0):
        _subid = _kill_list[0]
        is_mha = (_subid%2==0)
        _layer =layers[_subid//2]
        _layer.turn_off(mha=is_mha, mlp=not is_mha)
        # if is_mha:
        #     del _layer.input_layernorm
        #     del _layer.self_attn
        # else:
        #     del _layer.post_attention_layernorm
        #     del _layer.mlp
        del _kill_list [0]

    # Delete turn-offed layers
    for i in range(len(layers)-1, -1, -1):
        if layers[i].pass_layer:
            if _print:
                print(f"** prung {i}th layer")
            del layers[i]

    # Assign new layer_idx for attention heads
    _attn_idx = 0
    for i, layer in enumerate(layers):
        if not layer.pass_mha and not layer.pass_layer:
            layer.self_attn.layer_idx = _attn_idx
            _attn_idx += 1
    del layers
    clear_gpu_memory()
    return model

# [Section 3] GPU-related functions
def clear_gpu_memory():
    """
    Clearing the GPU memory
    """
    torch.cuda.empty_cache()
    gc.collect()

def get_device_map(
    layers: nn.ModuleList,
    num_checkpoints: int,
    num_cpu_layers:int,
    logger: logging.Logger):
    """
    Generate a device map and a checkpoint index map

    Args:
        layers (nn.ModuleList): layers
        num_checkpoints (int): the number of checkpoints
        num_cpu_layers (int): the number of layers on CPU
        logger (logging.Logger): a logger

    Returns:
        device_map (dict): device mapping for layers
        checkpoint_indices (list): checkpoint indices
    """
    
    device_map = {}
    num_layers_on_gpu = len(layers) - num_cpu_layers
    num_gpus = torch.cuda.device_count()
    nl = num_layers_on_gpu // num_gpus
    r = num_layers_on_gpu % num_gpus
    nums = [num_cpu_layers]
    _count = num_cpu_layers
    for i in range(num_cpu_layers):
        device_map[i] = 'cpu'

    for i in range(num_gpus):
        _device = f'cuda:{i}'
        for _ in range(nl):
            device_map[_count] = _device
            layers[_count].self_attn.rotary_emb.inv_freq = \
                layers[_count].self_attn.rotary_emb.inv_freq.to(_device)
            _count += 1
        if r > 0:
            device_map[_count] = _device
            layers[_count].self_attn.rotary_emb.inv_freq = \
                layers[_count].self_attn.rotary_emb.inv_freq.to(_device)
            _count += 1
            r -= 1
        nums.append(_count)

    if num_checkpoints == 0:
        checkpoint_indices = []
    else:
        checkpoint_step = len(layers) / (num_checkpoints+1)
        checkpoint_indices = [0] + [(int(_n * checkpoint_step)) * 2
                        for _n in range(1, num_checkpoints+1) ]
    
    return device_map, checkpoint_indices

def get_device(
    obj: Union[torch.Tensor, nn.Module]
) -> Union[str, torch.device]:
    """
    Get the device of a given obj (tensor of module)
    """

    if isinstance(obj, torch.Tensor):
        return obj.device
    return next(obj.parameters()).device

def get_memory_usage():
    """
    Return the amount of memory usage on GPUs
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_usage_gb = memory_info.rss / (1024**3)
    return memory_usage_gb


# [Section 4] miscellaneous
def create_logger(
    log_dir: str,
    name: str = '',
    log_filename:str = ''
):
    """
    Initialize logging file
    The implementation of this method is mainly based on:
        https://github.com/OpenGVLab/OmniQuant
    """

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Create formatter
    fmt = '[%(asctime)s %(name)s](%(filename)s %(lineno)d)|%(levelname)s| %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + \
                colored('|%(levelname)s|', 'blue') + ' %(message)s'

    # Create console handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S')
    )
    logger.addHandler(console_handler)

    # Create file handlers
    file_handler_debug = logging.FileHandler(
        os.path.join(log_dir, f'{log_filename}.log'), mode='a'
    )
    file_handler_debug.setLevel(logging.INFO)  # Change here to set level for log file
    file_handler_debug.setFormatter(
        logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S')
    )
    logger.addHandler(file_handler_debug)
    return logger

class UnImplementedError(Exception):
    """
    An exception class for unimplemented codes
    """
    def __str__(self):
        return "This code is not implemented, yet"
