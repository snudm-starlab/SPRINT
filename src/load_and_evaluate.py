"""
File: latency.py
- Load and evaluate pruned models
"""
import copy
import fire
import time
import torch
import os
from collections import OrderedDict

from glob import glob
from pathlib import Path
from torch import nn
from tqdm import tqdm

from utils.onoff_llama import convert
from utils.data_utils import *
from utils.eval_utils import comprehensive_eval
from utils.general_utils import *
from utils.compression_utils import *
from utils.CompressionManager import *

def method(
    # General
    model_name: str = 'meta-llama/Llama-2-7b-hf',
    log_dir: str = "./outputs/logs",
    log_file: str = "log",
    # Remove
    result_folder: str = './eval_results',
    result_file: str = 'results.txt',
    
    # Pruned model
    pruned_model_dir: str = './outputs/pruned_models/',
    pruned_model_file: str = 'Empty',
    pruned_sublayers: str = '',
    num_pruned_sublayers: int = 6,
    load_weight: bool = True,

    # For measuring latency
    generation: bool = True,
    generation_length: int = 512,
    prompt_length: int = 1024,
    batch_size: int = 1,
    iteration: int = 10,
):
    """
    Load and evaluate pruned models

    Args:
        model_name (str, optional): the name of a model
        log_dir (str, optional): a directry for saving logs. Defaults to "./outputs/logs".
        log_file (str, optional): a log file. Defaults to "log".
        pruned_model_dir (str, optional): the directory of the pruned model. Defaults to './outputs/pruned_models/'.
        pruned_model_file (str, optional): the file name of the pruned model. Defaults to 'Empty'.
        num_pruned_sublayers (int, optional): the number of sublayers to prune. Defaults to 6.
        load_weight (bool, optional): load the tuned weights or not. Defaults to True.
        generation (bool, optional): inference type (generation or summarization)
        generation_length (int, optional): the number of tokens to generate. Defaults to 1024.
        prompt_length (int, optional): the number of tokens in the prompt. Defaults to 512.
        batch_size (int, optional): batch size. Defaults to 1.
        iteration (int, optional): the number of iterations for latency measurement. Defaults to 10.
    """
    
    # generate log_file
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    os.makedirs(pruned_model_dir, exist_ok=True)
    log_dir = Path(log_dir)
    logger = create_logger(log_dir=log_dir, log_filename=log_file)
    
    pruned_model_path = os.path.join(pruned_model_dir, pruned_model_file)

    clear_gpu_memory()

    model = get_llm(model_name)

    use_cache = model.config.use_cache
    model.config.use_cache = False

    logger.info(f"* Loaded Model: {model.name}")
    logger.info(f"- Pruned file path: {pruned_model_path}")
    
    # Convert model into OnOffModel
    model = convert(model)
    model.eval()
    if pruned_model_file =="Empty":
        pruned_sublayers = list(pruned_sublayers)
        logger.info(f"- Pruned sublayers: {pruned_sublayers}")
        model = sublayer_remove(model, kill_list=pruned_sublayers)
    else:
        model, pruned_sublayers = load_pruned_model(model, 
                                                    pruned_model_path, 
                                                    num_remove=num_pruned_sublayers,
                                                    load_weight=load_weight)
    logger.info(f"* Prune {len(pruned_sublayers)} sublayers | IDs: {pruned_sublayers}")
    
    # Evaluate the model
    # _str = ''
    _str = comprehensive_eval(model)
    logger.info(f"\n* Evaluation results: {_str}")
    model.config.use_cache = use_cache

    # Measuring the latency of the model
    elf_latency = latency_utils.test_latency(
                        model,
                        generation,
                        generation_length,
                        prompt_length,
                        batch_size,
                        use_cache=True,
                        iteration=iteration,
                    )
    logger.info(f"* Latency: {elf_latency:.2f}ms")

if __name__ == "__main__":
    fire.Fire(method)
