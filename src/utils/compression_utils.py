"""
File: compression_utils.py
- Implementation of three main functions of SRPINT:
  latency approximation, candidate selection, and importance scoring
"""

import copy
import time
import torch
from tqdm import tqdm

from utils.CompressionManager import CompressionManager
from utils.onoff_llama import convert, turn_off, turn_on
from utils.general_utils import *
from utils import latency_utils

@torch.inference_mode()
def get_latency(model_name,
                remove_type,
                generation,
                generation_length,
                prompt_length,
                batch_size,
                iteration,
                ):
    """
    Estimating the latency of a model after pruning the given type of sublayers.

    Args:
        model_name: the name of the model
        remove_type: the type of sublayer to remove
        generation (bool, optional): inference type (generation or summarization)
        generation_length (int, optional): the number of tokens to generate. Defaults to 1024.
        prompt_length (int, optional): the number of tokens in the prompt. Defaults to 512.
        batch_size (int, optional): batch size. Defaults to 1.
        iteration (int, optional): the number of iterations for latency measurement. Defaults to 10.
        cache_dir (str, optional): cache directory path for saving the result. Defaults to '../cache'.

    Returns:
        float: the latency of the model with 25% pruned sublayers of a given type
    """
    # print(f"** Measuring for {remove_type} model start")
    latency_model = get_llm(model_name, device_map='auto')
    latency_model = convert(latency_model)
    latency_model.eval()
    layers = latency_model.model.layers
    num_layers = len(layers)
    num_remove = int(len(layers) * 0.25 + 1e-5)

    # Pruning sublayers according to remove_type
    if remove_type.lower() == 'mha':
        latency_model = sublayer_remove(latency_model, 
                            [2*(num_layers-2-i) for i in range(num_remove)])

        # Assign new layer_idx for attention heads
        _attn_idx = 0
        for i, layer in enumerate(layers):
            if not layer.pass_mha and not layer.pass_layer:
                layer.self_attn.layer_idx = _attn_idx
                _attn_idx += 1
        
    elif remove_type.lower() == 'mlp':
        latency_model = sublayer_remove(latency_model, 
                            [2*(num_layers-2-i)+1 for i in range(num_remove)])
    else:
        pass
    time.sleep(5) # cool down
    T = latency_utils.test_latency(
            latency_model,
            generation,
            generation_length,
            prompt_length,
            batch_size,
            use_cache=True,
            iteration=iteration,
    )
    latency_model.cpu()
    del latency_model, layers
    print(f"** Measuring for {remove_type} model done")
    clear_gpu_memory()
    return T
    

@torch.inference_mode()
def get_latencies(model_name,
                  generation:         bool = True,
                  generation_length:  int  = 1024,
                  prompt_length:      int  = 512,
                  batch_size:         int  = 1,
                  iteration:          int  = 10,
                  cache_dir:          str  = '../cache'
                  ):
    """
    Estimating latencies of MLP and MHA sublayers.

    Args:
        model_name: the name of the model
        generation (bool, optional): inference type (generation or summarization)
        generation_length (int, optional): the number of tokens to generate. Defaults to 1024.
        prompt_length (int, optional): the number of tokens in the prompt. Defaults to 512.
        batch_size (int, optional): batch size. Defaults to 1.
        iteration (int, optional): the number of iterations for latency measurement. Defaults to 10.
        cache_dir (str, optional): cache directory path for saving the result. Defaults to '../cache'.

    Returns:
        dict: a dictionary that contains latencies of sublayer 
    """
    
    # Initialization
    latency_model = get_llm(model_name, device_map='cpu')
    latency_model = convert(latency_model)
    latency_model.eval()

    layers = latency_model.model.layers
    num_remove = int(len(layers) * 0.25 + 1e-5)
    # Reduce the number of layers for efficiency
    latency_dict = {}
    
    model_type = latency_model.name.split("/")[-1]
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())

    env_key = '_'.join([str(_item) for _item in 
                        [gpu_name, model_type, generation, 
                         prompt_length, generation_length, batch_size]])
    
    lat_file = os.path.join(cache_dir, "latency.cache")
    
    if os.path.isfile(lat_file):
        cached_latency_dicts = torch.load(lat_file)
        if env_key in cached_latency_dicts:
            latency_dict = cached_latency_dicts[env_key]
            print("* Find and load the cached latency dict")
            del latency_model
            return latency_dict
    else:
        cached_latency_dicts = {}
    del latency_model, layers
    clear_gpu_memory()
    print("* Failed to find cached latenct dict. Start measuring.")

    # Mesuring T1: All layers
    st = time.time()
    
    print("Start measuring T1") # No gpu memory
    time.sleep(10) 
    T1 = get_latency(model_name,
                "unpruned",
                generation,
                generation_length,
                prompt_length,
                batch_size,
                iteration,
                )
    print("* End measuring T1")
    time.sleep(10)
    print("* Start measuring T2")
    T2 = get_latency(model_name,
                "MLP",
                generation,
                generation_length,
                prompt_length,
                batch_size,
                iteration,
                )
    time.sleep(10)
    T3 = get_latency(model_name,
                "MHA",
                generation,
                generation_length,
                prompt_length,
                batch_size,
                iteration,
                )
    print(f"* Total runnig time for measuring latencies: {time.time()-st:1f}s")
    
    # Compute latencies for MHA, MLP, and Residual (RES)
    latency_dict["MHA"] = (T1 - T3) / num_remove
    latency_dict["MLP"] = (T1 - T2) / num_remove
    
    print(f" * Latencies of [MHA/ MLP] = " + 
          f"[{latency_dict['MHA']:.2f}/ " + 
          f"{latency_dict['MLP']:.2f}]")
        
    cached_latency_dicts[env_key] = latency_dict
    torch.save(cached_latency_dicts, lat_file) # caching the latency dict
    return latency_dict


@torch.inference_mode()
def candidate_selection(
    model,
    start_subid:            int,
    chcekpoint_indices:     list,
    checkpoints:            dict,
    device_map:             dict,
    checkpointing_on_cpu:   bool,
    attention_mask_batch:   torch.Tensor,
    position_ids:           torch.Tensor,
    comp_manager:           CompressionManager,
    batch_size:             int                 = 4,
    tuning_sublayer:    str                 = 'MLP',
):
    """
    Selecting candidate sublayers by measuring importance scores without tuning

    Args:
        model: the target model to prune
        start_subid (int): the lowest sublayer index to measure importance score
        chcekpoint_indices (list): the indices of checkpoints
        checkpoints (dict): checkpoints
        device_map (dict): GPU device mapping
        checkpointing_on_cpu (bool): save cehckpoints on CPU or not
        attention_mask_batch (torch.Tensor): a batch of attention masks
        position_ids (torch.Tensor): position ids
        comp_manager (CompressionManager): compression manager
        batch_size (int, optional): batch size for importance scoring. Defaults to 4.
        tuning_sublayer (str, optional): the sublayer type for tuning. Defaults to 'MLP'.
    """

    model.eval()
    layers = model.model.layers
    num_sublayers = 2 * len(layers)

    # Remove cached inputs to update
    _del_ids = []
    for _subid in checkpoints.keys():
        if _subid > start_subid:
            _del_ids.append(_subid)
    for _subid in _del_ids:
        del checkpoints[_subid]

    clear_gpu_memory()

    # Get a checkpointed inputs
    if not checkpoints[start_subid].is_cuda: # cache in CPU
        if device_map[start_subid//2] == 'cpu':
            _device = 'cuda:0'
        else:
            _device = device_map[start_subid//2] 
        inps = checkpoints[start_subid].to(_device)
    else:
        inps = copy.deepcopy(checkpoints[start_subid])
    
    nsamples, _, _ = inps.shape

    # Switch all layers to half precision (for faster forward propagation)
    for layer in layers:
        layer.half()

    pbar = tqdm(range(start_subid, num_sublayers))
    for subid in pbar:
         # checkpoint inputs when `subid` is the target of caching
        if len(chcekpoint_indices) > 0 and (subid != start_subid) \
            and (subid in chcekpoint_indices):
            if checkpointing_on_cpu:
                checkpoints[subid] = inps.cpu()
            elif inps.device == get_device(layers[subid//2]):
                checkpoints[subid] = copy.deepcopy(inps)
            else:
                checkpoints[subid] = inps.to(get_device(layers[subid//2]))
            clear_gpu_memory()

        # Skip pruned sublayers
        if comp_manager.is_pruned(subid):
            pbar.set_description(f"* {subid}th sublayer is already Pruned")
            # print(f"* subid: {subid} | pruned -> SKIP")
            comp_manager.set_score_inf(subid)
            continue
        
        # Prepare sublayers to measure sensitivity
        comp_manager.reinit_imp_scores(subid)
        tuning_subid = get_tuning_subid(subid, 
                                                tuning_sublayer,
                                                comp_manager.pruning_status)
        
        # update tuning_indices
        comp_manager.tuning_indices[subid] = tuning_subid
        pbar.set_description(f"* Renewal sensitivity of {subid:3d} (using {tuning_subid:3d}'s output)")

        # Check and match devices
        inps, attention_mask_batch, position_ids = \
            check_and_match_devices(
                layers, 
                subid, 
                tuning_subid, 
                device_map, 
                inps, 
                attention_mask_batch, 
                position_ids,
                is_preprocess=True
                )
        
        # Measuring importance scores
        for i in range(nsamples // batch_size):
            # Get outputs of pruned model
            batch_comp = \
                batch_forward_multiple(
                    layers,
                    subid, tuning_subid, 
                    inps[i * batch_size: (i+1) * batch_size], 
                    attention_mask_batch, position_ids,
                    is_gt=False,
                    skip=True
                ) # compressed
            
            target_layer = layers[subid // 2]; target_is_mha = (subid % 2 == 0)
            if target_is_mha:
                target_module = target_layer.self_attn
                target_res_module = target_layer.input_layernorm
            else:
                target_module = target_layer.mlp
                target_res_module = target_layer.post_attention_layernorm

            new_inp = []; new_res = []
            inp_handle = hijack(target_module, 
                                new_inp, 
                                _hijack_input=False, 
                                _stop_forward=False, 
                                dtype=inps.dtype
                                )
            res_handle = hijack(target_res_module, 
                                new_res, 
                                _hijack_input=True, 
                                _stop_forward=False, 
                                dtype=inps.dtype)

            # Get outputs of unpruned model and update inputs
            batch_orig = \
                batch_forward_multiple(
                    layers,
                    subid, 
                    tuning_subid, 
                    inps[i * batch_size: (i+1) * batch_size], 
                    attention_mask_batch, 
                    position_ids,
                    is_gt=True,
                    skip=False,
                ) # origin
            
            # Update importance scores
            comp_manager.add_batch(subid, batch_orig, batch_comp)

            # Update `inps` for the next sublayer
            if target_is_mha:
                inps[i* batch_size: (i+1) * batch_size] = new_inp.pop(0)[0] + new_res.pop(0)
            else:
                inps[i* batch_size: (i+1) * batch_size] = new_inp.pop(0) + new_res.pop(0)
            inp_handle.remove()
            res_handle.remove()
            
        check_and_match_devices(
            layers, 
            subid, 
            tuning_subid, 
            device_map, 
            is_preprocess=False
            )
        clear_gpu_memory()
    del inps
    clear_gpu_memory()


@torch.inference_mode()
def importance_scoring(
    model,
    candidate_subids:       int,
    chcekpoint_indices:     list,
    checkpoints:            dict,
    device_map:             dict,
    tuned_weights:          dict,
    attention_mask_batch:   torch.Tensor,
    position_ids:           torch.Tensor,
    in_comp_manager:        CompressionManager,
    batch_size:             int                 = 4,
    tuning_sublayer:    str                 = 'MLP',
    in_comp_dtype:          torch.dtype         = torch.float32,
    in_comp_tuning_ratio:   float               = 0.1,
    direction:              str                 = 'out_channel',
    damping_coefficient:    float               = 0.
):
    """
    Measuring importance scores with in-compression tuning

    Args:
        model: the target model to prune 
        candidate_subids (int): the indices of candidate sublayer to measure the importance
        chcekpoint_indices (list): the indices of checkpoints
        checkpoints (dict): checkpoints
        device_map (dict): GPU device mapping
        tuned_weights (dict): a dictionary of sublayer indices and tuned weights
        attention_mask_batch (torch.Tensor): a batch of attention masks
        position_ids (torch.Tensor): position ids
        in_comp_manager (CompressionManager): compression_manager with in-compression tuning
        batch_size (int, optional): batch_size. Defaults to 4.
        tuning_sublayer (str, optional): the sublayer type for tuning. Defaults to 'MLP'.
        in_comp_dtype (torch.dtype, optional): the data type for in-compression tuning. Defaults to torch.float32.
        in_comp_tuning_ratio (float, optional): the ratio of weights for in-compression tuning. Defaults to 0.1.
        direction (str, optional): channel direction for selecting weights for tuning. Defaults to 'out_channel'.
        damping_coefficient (float, optional): damping coefficient for regularization. Defaults to 0..

    Returns:
        tuned_weights (dict): an updated tuned_weights
    """

    model.eval()
    layers = model.model.layers
    min_subid = min(candidate_subids)
    max_subid = max(candidate_subids)
    if len(chcekpoint_indices) > 0:
        i = -1
        while chcekpoint_indices[i] > min_subid:
            i -= 1
        start_subid = chcekpoint_indices[i]
    else:
        start_subid = 0
    end_subid = max_subid

    # Load a checkpointed input
    if checkpoints[start_subid].device == 'cpu':
        if device_map[start_subid//2] == 'cpu':
            _device = 'cuda:0'
        else:
            _device = device_map[start_subid//2] 
        inps = checkpoints[start_subid].to(_device)
    else:
        inps = copy.deepcopy(checkpoints[start_subid])
    nsamples, _, _ = inps.shape

    # Switch all layers to half precision for faster forward propagation
    for layer in layers:
        layer.half()
    
    pbar = tqdm(range(start_subid, end_subid+1))
    for subid in pbar:
        # Skip pruned sublayers
        if in_comp_manager.is_pruned(subid):
            pbar.set_description(f"* {subid:3d}th sublayer is already pruned")
            in_comp_manager.set_score_inf(subid)
            continue
        
        # Prepare sublayers to measure sensitivity
        if subid in candidate_subids:
            pbar.set_description(f"* Subid: {subid:3d}")
            tuning_subid = get_tuning_subid(subid, 
                                                    tuning_sublayer,
                                                    in_comp_manager.pruning_status)
            in_comp_manager.reinit_imp_scores(subid)  
        else:
            pbar.set_description(f"* Subid: {subid:3d}")
            tuning_subid = subid

        # Check and match devices
        inps, attention_mask_batch, position_ids = \
            check_and_match_devices(layers, 
                                    subid, 
                                    tuning_subid, 
                                    device_map, 
                                    inps, 
                                    attention_mask_batch, 
                                    position_ids,
                                    is_preprocess=True)    
        
        if (subid == tuning_subid) and (subid in candidate_subids):
            # the sublayer is already pruned and cannot perform in_comp_tuning
            in_comp_manager.set_score_inf(subid)
            print(f"subid ({subid}) is equal to the look forward_subid ({tuning_subid}).",
                  "Cannot perform in-compression tuning and skip")
            continue

        if subid in candidate_subids:
            _tuned_weights = {}
            
            #  (1) Perform in-compression tuning for output projection
            #  We have to use tuned weights for generating pruned model's output
            #  and use original weights for generating original model's output
            _tuned_weights, tuned_out_weight, \
            batch_orig_outputs, batch_comp_outputs = \
                out_proj_tuning(layers, 
                                subid, 
                                tuning_subid, 
                                inps, 
                                attention_mask_batch,
                                position_ids,
                                in_comp_tuning_ratio, 
                                batch_size, 
                                in_comp_dtype,
                                _tuned_weights,
                                direction,
                                damping_coefficient)
            tuned_weights[subid] = (tuning_subid, _tuned_weights)
        
            # (2) Update importance scores and update inps
            # Compute importance scores
            _device = tuned_out_weight.device
            for _origs, _comps in zip(batch_orig_outputs, batch_comp_outputs):
                batch_orig = _origs.to(_device)
                batch_comp = _comps[0].to(_device) @ (tuned_out_weight.T) + _comps[1].to(_device) 
                in_comp_manager.add_batch(subid, batch_orig, batch_comp)
                    
            del batch_orig_outputs, batch_comp_outputs
            clear_gpu_memory()

        else:
            # Forwarding wihtout tuning for non-candidate sublayers
            for i in range(nsamples // batch_size):
                inps[i * batch_size: (i+1) * batch_size] = \
                                        batch_forward_multiple(
                                            layers,
                                            subid, subid, 
                                            inps[i * batch_size: (i+1) * batch_size], 
                                            attention_mask_batch, position_ids,
                                            is_gt=True,
                                            skip=False,
                                        ) # origin

            
        check_and_match_devices(layers, 
                                subid, 
                                tuning_subid, 
                                device_map, 
                                is_preprocess=False)
        clear_gpu_memory()

    del inps
    clear_gpu_memory()
    return tuned_weights

def out_proj_tuning(layers, 
                    sid:                    int, 
                    tuning_sid:         int, 
                    inps:                   torch.Tensor, 
                    attention_mask_batch:   torch.Tensor, 
                    position_ids:           torch.Tensor,
                    channel_ratio:          float,
                    batch_size:             int, 
                    dtype:                  torch.dtype,
                    tuned_weights:          dict,
                    direction:              str             ='in_channel',
                    damping_coefficient:    float           =0.):
    """
    in-compression tuning that tunes the output projection of sublayers

    Args:
        layers: the layers of the model
        sid (int): the sublayer index for measuring importance
        tuning_sid (int): the sublayer index for tuning
        inps (torch.Tensor): input activations of sid-th sublayer
        attention_mask_batch (torch.Tensor): a batch of attention masks
        position_ids (torch.Tensor): position ids
        channel_ratio (float): the ratio of weights for in-compression tuning
        batch_size (int): batch_size
        dtype (torch.dtype): the data type for in-compression tuning.
        tuned_weights (dict): a dictionary of sublayer indices and tuned weights
        direction (str, optional): channel direction for selecting weights for tuning.. Defaults to 'in_channel'.
        damping_coefficient (float, optional): damping coefficient for regularization. Defaults to 0..

    Returns:
       tuned_weights: a tuned weight and its index
       tuned_out_weight: the tuned weight of the output projection for importance scoring
       batch_orig_outputs: a batch of activations of the unpruned model for importance scoring
       batch_comp_outputs: a batch of activations of the pruned model for importance scoring
    """
    
    # Initialize
    X_out_pruned = []; X_out_pruned_res = []
    tuning_layer = layers[tuning_sid//2]
    out_proj = get_output_proj(layers, tuning_sid)
    _ln = get_layernorm(layers, tuning_sid)
    d_out, d_in = out_proj.weight.shape

    # Getting channel_mask
    # Initialization
    Xsq = None; pruned_inputs = []
    orig_weights = {}
    assert tuning_sid % 2 == 1, "Only supports MLP-tuing"
            
    if 'up_proj' in tuned_weights:
        orig_weights['up_proj'] = copy.deepcopy(tuning_layer.mlp.up_proj.weight)
        tuning_layer.mlp.up_proj.weight.data[tuned_weights['up_proj'][1], :] = \
        tuned_weights['up_proj'][0]

    if 'gate_proj' in tuned_weights:
        orig_weights['gate_proj'] = copy.deepcopy(tuning_layer.mlp.gate_proj.weight)
        tuning_layer.mlp.gate_proj.weight.data[tuned_weights['gate_proj'][1], :] = \
            tuned_weights['gate_proj'][0]
    
    p_handle = hijack(out_proj, 
                      pruned_inputs,
                      _hijack_input=True,
                      _stop_forward=True,
                      dtype=dtype)
    # (1) Get pruned model's outputs   
    for i in range(inps.shape[0]//batch_size):
        try:
            batch_forward_multiple(
                layers,
                sid, 
                tuning_sid, 
                inps[i * batch_size: (i+1) * batch_size],
                attention_mask_batch, position_ids,
                is_gt=False,
                skip=True)
        except StopForwardException:
            pass
        Xp = pruned_inputs.pop(0).view(-1, d_in)
        
        assert Xp.dtype == dtype
        
        # Update Xsq for channel selection
        if Xsq is None:
            Xsq = (Xp*Xp).sum(dim=0) # [d_in]
        else:
            Xsq += (Xp*Xp).sum(dim=0)
    p_handle.remove()
    # (2) Select channels to tune
    if direction == "out_channel":
        channel_score = (out_proj.weight.abs().data * Xsq.sqrt().view(1, -1)).sum(dim=1)
        # True: tune | False: no tune
        column_mask = (channel_score >= \
                    channel_score.topk(int(d_out * channel_ratio)).values[-1])
    elif direction == "in_channel":
        channel_score = (out_proj.weight.abs().data * Xsq.sqrt().view(1, -1)).sum(dim=0)
        # True: tune | False: no tune
        column_mask = (channel_score >= \
                    channel_score.topk(int(d_in * channel_ratio)).values[-1])
    else:
        raise Exception("Unkwon channel direction")


    XpTXp = None; XpTX=None
    batch_orig_outputs = []; batch_comp_outputs = []    
    
    # (3) Forwarding and solving lstsq
    for i in range(inps.shape[0] // batch_size):
        if 'up_proj' in tuned_weights:
            tuning_layer.mlp.up_proj.weight.data[tuned_weights['up_proj'][1], :] = \
                tuned_weights['up_proj'][0]
        if 'gate_proj' in tuned_weights:
            tuning_layer.mlp.gate_proj.weight.data[tuned_weights['gate_proj'][1], :] = \
                tuned_weights['gate_proj'][0]

        tuning_inp_handle = hijack(out_proj, X_out_pruned, 
                               _hijack_input=True, 
                               _stop_forward=True, 
                               dtype=torch.float16)
        tuning_res_handle = hijack(_ln, X_out_pruned_res, 
                                _hijack_input=True, 
                               _stop_forward=False, 
                               dtype=torch.float16)
        # Compute pruned model's output
        try:
            batch_forward_multiple(
                layers,
                sid, tuning_sid, 
                inps[i * batch_size: (i+1) * batch_size],
                attention_mask_batch, position_ids,
                is_gt=False,
                skip=True)
        except StopForwardException:
            pass
        tuning_inp_handle.remove()
        tuning_res_handle.remove()
        
        # Get unpruned outputs and Update inps
        target_layer = layers[sid // 2]; target_is_mha = (sid % 2 == 0)
        if target_is_mha:
            target_module = target_layer.self_attn
            target_res_module = target_layer.input_layernorm
        else:
            target_module = target_layer.mlp
            target_res_module = target_layer.post_attention_layernorm

        new_inp = []; new_res = []
        inp_handle = hijack(target_module, new_inp, 
                            _hijack_input=False, _stop_forward=False, 
                            dtype=inps.dtype)
        res_handle = hijack(target_res_module, new_res, 
                            _hijack_input=True, _stop_forward=False, 
                            dtype=inps.dtype)
        
        if 'up_proj' in tuned_weights:
            tuning_layer.mlp.up_proj.weight.data = orig_weights['up_proj']
        if 'gate_proj' in tuned_weights:
            tuning_layer.mlp.gate_proj.weight.data = orig_weights['gate_proj']

        _X_orig = \
            batch_forward_multiple(
                layers,
                sid, tuning_sid, 
                inps[i * batch_size: (i+1) * batch_size], 
                attention_mask_batch, position_ids,
                is_gt=True,
                skip=False,
            ) # origin

        # Update `inps` for the next sublayer
        if target_is_mha:
            inps[i* batch_size: (i+1) * batch_size] = new_inp.pop(0)[0] + new_res.pop(0)
        else:
            inps[i* batch_size: (i+1) * batch_size] = new_inp.pop(0) + new_res.pop(0)
        inp_handle.remove()
        res_handle.remove()
        
        # Update XpTXp and XpTX to solve a least square problem
        _X_out_pruned = X_out_pruned.pop()
        _X_out_pruned_res = X_out_pruned_res.pop()

        batch_orig_outputs.append(_X_orig)
        batch_comp_outputs.append((_X_out_pruned, _X_out_pruned_res))
        if direction == 'out_channel':
            fp_X_orig = _X_orig.view(-1, d_out)[:, column_mask].to(dtype=dtype)
            fp_X_out_pruned = _X_out_pruned.view(-1, d_in).to(dtype=dtype)
            fp_X_out_pruned_res = _X_out_pruned_res.view(-1, d_out)[:, column_mask].to(dtype=dtype)
            pass
        elif direction == 'in_channel':
            fp_X_orig = _X_orig.view(-1, d_out).to(dtype=dtype)
            fp_X_out_pruned = _X_out_pruned.view(-1, d_in)[:, column_mask].to(dtype=dtype)
            fp_X_out_pruned_res = _X_out_pruned_res.view(-1, d_out).to(dtype=dtype)
            pass
        else:
            raise Exception("Unkwon channel direction")

        if XpTXp is None:
            XpTXp = fp_X_out_pruned.T @ fp_X_out_pruned # [d_in,ns] @ [ns,d_in]
            XpTX = fp_X_out_pruned.T @ (fp_X_orig-fp_X_out_pruned_res) # [d_in,ns] @ [ns,d_out]
        else:
            XpTXp += fp_X_out_pruned.T @ fp_X_out_pruned
            XpTX += fp_X_out_pruned.T @ (fp_X_orig-fp_X_out_pruned_res)
        del fp_X_out_pruned, fp_X_out_pruned_res, fp_X_orig
        clear_gpu_memory()
    
    orig_weight = out_proj.weight.data
    d_out, d_in = orig_weight.shape
    if damping_coefficient <= 0.:
        ATA = XpTXp; ATB = XpTX # [d_in, d_in], [d_in, d_out]
    else:
        ATA = torch.cat([XpTXp, 
                            damping_coefficient * torch.eye(d_in)\
                            .to(device=XpTXp.device, dtype=XpTXp.dtype)])
        ATB = torch.cat([XpTX, 
                            damping_coefficient * copy.deepcopy(orig_weight.data.T)\
                            .to(device=XpTX.device, dtype=XpTX.dtype)])
    

    _tuned_weight = torch.linalg.lstsq(ATA, ATB).solution.T.contiguous().half()
    if _tuned_weight.isnan().sum() > 0 or (_tuned_weight==float('inf')).sum():
        # Drop the tuned weight when nan is in it
        print("Failed in-compression tuning. Find nan or inf. Drop the tuning results")
        if direction == "in_channel":
            _tuned_weight = copy.deepcopy(orig_weight[:,column_mask])
        elif direction == 'out_channel':
            _tuned_weight = copy.deepcopy(orig_weight[column_mask,:])
        else:
            raise Exception("Unkwon channel direction") 

    if tuning_sid % 2 ==0:
        tuned_weights['o_proj'] = (_tuned_weight, column_mask, direction)
    else:
        tuned_weights['down_proj'] = (_tuned_weight, column_mask, direction)
    
    # Return tuned_out_weight for importance measuring
    tuned_out_weight = copy.deepcopy(orig_weight)
    if direction == "in_channel":
        tuned_out_weight[:, column_mask] = _tuned_weight
    elif direction == 'out_channel':
        tuned_out_weight[column_mask, :] = _tuned_weight
    else:
        raise Exception("Unkwon channel direction") 
    
    del ATA, ATB, orig_weights
    clear_gpu_memory()
    return tuned_weights, tuned_out_weight, batch_orig_outputs, batch_comp_outputs
