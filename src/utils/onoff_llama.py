"""
File: onoff.py
- An implementation of OnOff_LlamaDecoderLayer for sublayer pruning
- This source code is written based on the following GitHub repository:
    https://github.com/jiwonsong-dev/SLEB
"""

import torch
import torch.nn as nn
from utils.general_utils import * 
from typing import List, Optional, Tuple
import warnings

class OnOff_LlamaDecoderLayer(nn.Module):
    def __init__(self, original_decoder_layer):
        super().__init__()
        self.hidden_size = original_decoder_layer.hidden_size
        self.self_attn = original_decoder_layer.self_attn
        self.mlp = original_decoder_layer.mlp
        self.input_layernorm = original_decoder_layer.input_layernorm
        self.post_attention_layernorm = original_decoder_layer.post_attention_layernorm

        self.pass_layer = False
        self.pass_mha = False
        self.pass_mlp = False

    def turn_off(self, mha=True, mlp=True):
        """
        Turn off either an MHA or an MLP sublayer
        """
        if mha:
            self.pass_mha = True
        if mlp:
            self.pass_mlp = True
        self.pass_layer = (self.pass_mha and self.pass_mlp)
    
    def turn_on(self, mha=True, mlp=True):
        """
        Turn on either an MHA or an MLP sublayer
        """
        if mha:
            self.pass_mha = False
        if mlp:
            self.pass_mlp = False
        self.pass_layer = (self.pass_mha and self.pass_mlp)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        The forward function of OnOff_LlamaDecoderLayer

        Args:
            hidden_states (torch.Tensor): a hidden state
            attention_mask (Optional[torch.Tensor], optional): an attention mask. Defaults to None.
            position_ids (Optional[torch.LongTensor], optional): a position ids. Defaults to None.
            past_key_value (Optional[Tuple[torch.Tensor]], optional): past key and values. Defaults to None.
            output_attentions (Optional[bool], optional): Whether or not to return the attentions tensors of all attention layers.. Defaults to False.
            use_cache (Optional[bool], optional): use KV cache or not. Defaults to False.

        Returns:
            outputs: a tuple of outputs
        """
        # skip this decoder layer
        if self.pass_layer:
            outputs = (hidden_states,)

            if output_attentions:
                outputs += (None,)

            if use_cache:
                outputs += (past_key_value,)

            return outputs
        
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        # Self Attention
        if self.pass_mha:
            # skipping mha
            self_attn_weights = None
            present_key_value = past_key_value
        else:
            hidden_states, attention_mask, position_ids = \
            move_inputs(hidden_states, attention_mask, position_ids, self.self_attn)
            residual = hidden_states

            hidden_states = self.input_layernorm(hidden_states)

            
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
            if residual.device != hidden_states.device:
                residual = residual.to(hidden_states.device)

            hidden_states = residual + hidden_states

        # Fully Connected
        if self.pass_mlp:
            pass
        else:
            hidden_states, attention_mask, position_ids = \
            move_inputs(hidden_states, attention_mask, position_ids, self.mlp)
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)

            if residual.device != hidden_states.device:
                residual = residual.to(hidden_states.device)

            hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    def do_mlp_forward(self, hidden_states):
        """
        A forward function for MLP sublayers
        """
        if self.pass_mlp:
            return hidden_states
        # Align devices of inputs and weights before computating
        hidden_states, _, _ = move_inputs(
            hidden_states, None, None, self
        )

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if residual.device != hidden_states.device:
            residual = residual.to(hidden_states.device)

        hidden_states = residual + hidden_states
        return hidden_states
        

    def do_mha_forward(self, hidden_states, attention_mask=None, position_ids=None,
                       past_key_value=None, output_attentions=False, use_cache=False):
        """
        A forward function for MHA sublayers
        """
        if self.pass_mha:
            return hidden_states, None, None
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Align devices of inputs and weights before computating
        hidden_states, attention_mask, position_ids = move_inputs(
            hidden_states, attention_mask, position_ids, self
        )
        
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        if residual.device != hidden_states.device:
            residual = residual.to(hidden_states.device)

        hidden_states = residual + hidden_states
        return hidden_states, self_attn_weights, present_key_value

def convert(model):
    """
    Convert the layers into a model into OnOff_LlamaDecoderLayers
    """
    num_layers = len(model.model.layers)
    for i in range(num_layers):
        model.model.layers[i] = OnOff_LlamaDecoderLayer(model.model.layers[i])
    return model

def turn_off(model, sublayer_idx):
    """
    Turn off the target sublayer
    """
    block_idx = sublayer_idx // 2
    is_mha = (sublayer_idx%2 == 0)
    model.model.layers[block_idx].turn_off(mha=is_mha, mlp=not is_mha)

def turn_on(model, sublayer_idx):
    """
    Turn on the target sublayer
    """
    block_idx = sublayer_idx // 2
    is_mha = (sublayer_idx%2 == 0)
    model.model.layers[block_idx].turn_on(mha=is_mha, mlp=not is_mha)
    


