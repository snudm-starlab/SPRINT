"""
File: latency_utils.py
- Utility functions for measuring the latency of models
- This source code is written based on the following GitHub repository:
    https://github.com/jiwonsong-dev/SLEB
"""

import torch
import accelerate

@torch.no_grad()
def test_latency(
    model,
    generation,
    generation_length=512,
    prompt_length=1024,
    batch_size=1,
    use_cache=True,
    iteration=10,
) :
    """
    Measuring the latency of models

    Args:
        model (_type_): the name of the model
        generation (bool, optional): inference type (generation or summarization)
        generation_length (int, optional): the number of tokens to generate. Defaults to 1024.
        prompt_length (int, optional): the number of tokens in the prompt. Defaults to 512.
        batch_size (int, optional): batch size. Defaults to 1.
        use_cache (bool, optional): use KV cache or not. Defaults to True.
        iteration (int, optional): the number of iterations for latency measurement. Defaults to 10.

    Returns:
        mean_latency: the measured latency
    """

    if (generation) :
        # setting for token generation
        max_length = prompt_length + generation_length
        model.config.max_length = max_length
        model.config.use_cache = use_cache
        model.generation_config.use_cache = use_cache
        # iteration = iteration

        # make dummy input
        random_input = torch.randint(0, 31999, (batch_size, prompt_length), dtype=torch.long)
        random_input = random_input.to(model.device).contiguous()
        _pad_token_id = model.generation_config.eos_token_id if \
            not hasattr(model.generation_config, 'pas_token_id') else model.generation_config.pas_token_id
        # dummy inference
        model.generate(random_input,
                       attention_mask=torch.ones_like(random_input),
                       pad_token_id=_pad_token_id,
                       min_new_tokens=generation_length, 
                       max_new_tokens=generation_length)
        
        # latency for 10 iterations
        starter,ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)

        starter.record()
        for i in range(iteration):
            model.generate(random_input,
                           attention_mask=torch.ones_like(random_input),
                           pad_token_id=_pad_token_id,
                           min_new_tokens=generation_length,
                           max_new_tokens=generation_length,
                           )
        ender.record()
        torch.cuda.synchronize()

    else :
        # setting for prompt processing
        batch_size = 1
        model.config.use_cache = False
        model.generation_config.use_cache = False

        # make dummy input for module.weight shape
        random_input = torch.randint(0, 31999, (batch_size, 2048), dtype=torch.long)
        random_input = random_input.to(model.device).contiguous()
        
        # dummy inference
        model(random_input)

        # latency for 50 iterations
        starter,ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        starter.record()
        for i in range(iteration):
            model(random_input)
        ender.record()
        torch.cuda.synchronize()

    curr_time = starter.elapsed_time(ender)
    mean_latency = curr_time/iteration

    return mean_latency
