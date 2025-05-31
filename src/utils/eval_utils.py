"""
File: eval_utils.py
- Utility functions to evalutae models
- We support both perplexity and CSR benchmarks
- This source code is written based on the following GitHub repository:
    https://github.com/jiwonsong-dev/SLEB
"""

import os
import torch
from torch import nn
from tqdm import tqdm
from typing import Union
from utils.general_utils import *
from utils.data_utils import *

# Function to evaluate perplexity (ppl) on a specified model and tokenizer
@torch.no_grad()
def load_and_eval_ppl(
    model,
    device: Union[str, torch.device] = torch.device("cuda:0"),
    dataset: str = "wikitext2",
    testloader = None,
    tokenizer = None
): 
    """
    Evaluate the ppl of models
    """
    
    # Print status
    print(f"Evaluating on {dataset}")
    
    # Get the test loader
    if testloader is None:
        if tokenizer is None:
            tokenizer = get_tokenizer(model.name)

        _, testloader = get_loaders(
            dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer 
        )
        print(f"Dataset Loaded.")

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl_test = eval_ppl(model, testloader, 1, device)
    return ppl_test 

@torch.no_grad()
def eval_ppl(model, testenc, bs=1, device=None):
    """
    Evaluate the ppl of models
    """
    # Check all modules are on `device`
    outside_layer_modules = ["model.embed_tokens", "model.norm", "lm_head"] # TODO: llama-specific
    for name, module in model.named_modules():
        if name in outside_layer_modules:
            if not module.weight.data.is_cuda:
                print(f"Move {name} to {device}")
                module = module.to(device)
    for layer in model.model.layers:
        if not layer.self_attn.q_proj.weight.data.is_cuda:
            print(f"Move a layer to {device}")
            layer = layer.to(device)
    
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in tqdm(range(0, nsamples, bs)):

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)


        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous().to(device)
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    return ppl.item()

@torch.no_grad()
def eval_zero_shot(task_list=['piqa','hellaswag','arc_challenge','arc_easy','boolq'], 
                   num_fewshot=0, 
                   compressed_model=None,
                   ):
    """
    Evaluate the accuracy of models on CommonSense Reasoning (CSR) benchmarks
    """
    
    from lm_evaluation_harness.lm_eval import tasks, evaluator, utils
    task_manager = tasks.TaskManager(include_path='lm-evaluation-harness/lm_eval/tasks')
 
    task_names = task_manager.match_tasks(task_list)
    for task in [task for task in task_list if task not in task_names]:
                if os.path.isfile(task):
                    config = utils.load_yaml_config(task)
                    task_names.append(config)
    task_missing = [
        task
        for task in task_list
        if task not in task_names and "*" not in task
        ]  # we don't want errors if a wildcard ("*") task name was used
    
    model_args = ''

    results = evaluator.simple_evaluate(
        model='hf',
        model_args=model_args,
        tasks=task_list,
        num_fewshot=num_fewshot,
        batch_size=8,
        max_batch_size=None,
        device='auto',
        use_cache=None,
        limit=None,
        decontamination_ngrams_path=None,
        check_integrity=False,
        write_out=False,
        gen_kwargs=None,
        task_manager=task_manager,
        verbosity = "ERROR",
        compressed_model=compressed_model,
    )

    return results 

@torch.inference_mode()
def comprehensive_eval(model):
    """
    Evaluate the ppls and accuracies on various benchmarks

    Args:
        model: a model to evaluate

    Returns:
        _str (str): a string of evaluation results
    """
    _eval_count = 0
    _str = ''
    while _eval_count < 900:
        try:
            w2_ppl = load_and_eval_ppl(model, device=torch.device("cuda:0"), dataset='wikitext2')
            _eval_count = 999
        except Exception as e:
            print("[Faild to eval ppl] ", e)
            _eval_count += 1 
    print(f"Starting Zero-shot tasks evaluation...")
    _str += f'{w2_ppl:.4f}'
    tasks = ['piqa','hellaswag','arc_challenge','arc_easy','boolq']
    
    results = eval_zero_shot(tasks, 
                             compressed_model=model)
    results = results['results']
    _tot_acc = 0.; _count = 0
    print("==" * 45)
    print(f"* Evaluation results")
    print(f" - WikiText2 ppl: {w2_ppl:.2f}")
    print(f" - Zero-shot accuracies")
    
    for task in tasks:
        if 'acc_norm,none' in results[task]:
            _acc = results[task]['acc_norm,none']
        else:
            _acc = results[task]['acc,none']
        _str += f",{_acc*100:.4f}"
        print(f"  {task}: {_acc*100:.4f}")
        _tot_acc += _acc
        _count += 1
    _str += f",{_tot_acc/_count*100:.4f}"
    print(f"  Average: {_tot_acc/_count*100:.4f}")
    print("==" * 45)
    return _str