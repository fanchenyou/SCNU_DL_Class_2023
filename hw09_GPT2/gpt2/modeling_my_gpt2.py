# coding=utf-8

# original src https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""Revision of PyTorch OpenAI GPT-2 model for teaching purpose."""

import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import torch
import torch.utils.checkpoint
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import Conv1D
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

_CHECKPOINT_FOR_DOC = "gpt2"
_CONFIG_FOR_DOC = "GPT2Config"

GPT2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "gpt2",
]


class GPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()

        max_positions = config.max_position_embeddings

        # In our simplified version, we won't use this two variables.
        # put here as dummy lines to avoid code warning
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.layer_idx = layer_idx

        # TODO: Explain why c_attn has 3*embed_dim input size ?
        # Hint: check its usage below. Also use pycharm/vscode to jump to Conv1D declaration,
        # should be in Transformers/pytorch_utils.py
        self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _attn(self, query, key, value, attention_mask=None, use_cache=False):

        query_length, key_length = query.size(-2), key.size(-2)


        # Q * K'
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        # QK' / sqrt(d)
        attn_weights = attn_weights / (value.size(-1) ** 0.5)

        ##########################################
        ## The code below is the core of att #####
        ##########################################
        if use_cache and query_length == 1:
            # use_cache has two cases
            # case 1: the first round, the entire prompt is feed into decoder, query_length>1, thus go to "else"
            # case 2: the following rounds, generating next-token word-by-word, query_length==1, thus we pass, do nothing
            pass
        else:
            # TODO: explain the masked attention
            # read lecture note and explain why adding "-inf" to mask and add to attention
            mask = torch.full(
                (1, 1, query_length, query_length), float("-inf"), device=query.device
            )
            attention_mask = torch.triu(mask, diagonal=1).type_as(query)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask
            # print(attention_mask)

        # TODO: briefly explain the operations, is it standard attention formula ?
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:

        # transform hidden_states to Q,K,V
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        # split to multi-head attention
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # TODO: Explain the function of layer_past, check where this function is called
        # hint: is this from KV_cache?
        if layer_past is not None:
            past_key, past_value = layer_past
            assert len(layer_past)==2
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            kv_tensor = (key, value)

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, use_cache)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        if use_cache:
            return attn_output, kv_tensor

        return attn_output, None


class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPT2Block(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(inner_dim, config)

    def forward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            use_cache=use_cache,
        )

        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        kv_tensor = attn_outputs[1:]

        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        # TODO: explain why we need to append outputs to Block output
        # hint: check where is this called and returned, check GPT2Model-> forward() below
        if use_cache:
            # need to pass present in outputs
            out = (hidden_states,) + kv_tensor
        else:
            # do not need to pass present
            # only pass current hidden state
            out = (hidden_states,)  # + outputs[1:]

        return out  # hidden_states, present, (attentions, cross_attentions)


class GPT2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPT2Config
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPT2Block"]
    _skip_keys_device_placement = "past_key_values"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)


class GPT2Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        # word embedding
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        # positional embedding
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]

        ########################################
        ### This is the realistic Cache   ######
        ########################################
        # for no_use_cache, always None
        # for use_cache, the first time run is None, then growing
        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))
        else:
            # you can print out the Cache, it's a 12-layer of (key, value)
            # each key and value of size [1, 12, L, 64] in each loop
            # for pkv in past_key_values:
            #     print(pkv[0].size(), pkv[1].size())
            pass

        # make attention_mask [1, L] -> [1, 1, 1, L]
        attention_mask = attention_mask[:, None, None, :]

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)

        # TODO: Explain, why adding input_embeds with PEs
        # check Transformer paper, https://arxiv.org/pdf/1706.03762.pdf
        # read section 3.5
        hidden_states = inputs_embeds + position_embeds

        hidden_states = self.drop(hidden_states)

        cur_key_values = () if use_cache else None

        ###################################################################
        # TODO: Understand KV-cache passes in previous K and V to attention
        ###################################################################
        # layer_past passes in K, V cache data
        # check GPTAttention -> forward() -> line "past_key, past_value = layer_past"
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            # check the GPT2Block output
            hidden_states = outputs[0]

            ################################################
            ### TODO: Explain what is cur_key_values ?

            ###############################################
            # hint: stack kv_cache over all attention layers
            # is this new kvs that need to pass to next loop in generation loop ?
            # uncomment the print line, check if KV tensor is gradually expanding as [B, N_head, L, D]

            if use_cache is True:
                kv_tensor = outputs[1]
                cur_key_values = cur_key_values + (kv_tensor,)
                # k,v = kv_tensor
                # print(k.size(),v.size())

        hidden_states = self.ln_f(hidden_states)

        # return past_key_values to greedy_search step
        # in greedy_search, it's stored in model_kwargs["past_key_values"]
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=cur_key_values,
        )


################################################################
### As Language Modeling, we have word prediction as output ####
################################################################
class GPT2LMHeadModel(GPT2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

        print('This is a modified GPT2 implementation for teaching purpose')
        print(config)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):

        #########################
        #### TODO: Understand ###
        #########################
        # if use_cache: only last token for inputs_ids 
        # this is because previous tokens have been cached as K and V in past_key_values
        # if not use_cache, you have to put in all previous token_ids
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        #########################
        #### TODO: Understand ###
        #########################
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = None

        # here generate attention mask
        # if use_cache, position is [L], in which L is the current length of the sentence
        # if not use_cache, position is [0,1,2,3,...,L-1]
        # use for positional embedding, check line "position_embeds = self.wpe(position_ids)"
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        #########################
        #### TODO: Understand ###
        #########################
        # the real input for each loop of greedy search
        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": None,
        }

        return model_inputs

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # check GPT2Model forward()
        # this produces hidden outputs for each loop in Main Generation Loop
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # extract hidden state for each loop
        hidden_states = transformer_outputs[0]
        # use linear layer to decode a prob. distribution over all words in vocabulary, i.e., decoding to a word
        lm_logits = self.lm_head(hidden_states)

        ## This is used in Training, return loss if provided labels
        ## It's shifted prediction with CE loss
        ## NOT used in inference. Just read about it.
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        ## return dictionary as output. go back to greedy search.
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    ############################################
    ### This is the main entry
    ### Let's call this Main Generation Loop ###
    ############################################
    def generate_greedy_search(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            eos_token_id_tensor: torch.LongTensor,
            max_length: Optional[int] = None,
            use_cache: Optional[bool] = None
    ) -> tuple[Tensor, Tensor]:

        # for simplicity, we only support one sentence at a time, i.e., batch_size==1
        batch_size = input_ids.size(0)
        assert batch_size == 1

        # we initialize the logprob to 0
        # then accumulate logprob of each word in batch_scores in loop below
        batch_scores = torch.zeros(batch_size, dtype=torch.float, device=input_ids.device)

        # initialize an empty KV_cache
        # will add past_key_values in while loop
        KV_cache = {}
        KV_cache['use_cache'] = use_cache
        KV_cache['attention_mask'] = attention_mask

        # let's check the input_ids
        # "I enjoy walking with" has ids  tensor([[   40, 2883, 3375,  351]])
        # lookup vocab.json, you find No.40 - "I", and so on.
        # check and print the other words in tokenizer vocabulary

        while True:
            # prepare model inputs
            # this function is implemented by modeling_my_gpt2.py -> GPT2LMHeadModel -> prepare_inputs_for_generation()
            # check what it functions
            model_inputs = self.prepare_inputs_for_generation(input_ids, **KV_cache)

            ###################################
            # TODO-Explain
            # forward pass to get next token
            # check GPT2LMHeadModel->forward()
            ###################################
            # In greedy search, each step calls GPT2LMHeadModel forward() 
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )
            # the above step is for next token prediction

            # the output logits of all tokens in vocabulary
            # for use_cache, outputs.logits is size [1, 1, 50257], since it generates only the last token
            # for not_use_cache, outputs.logits is size [1, L, 50257], since it generates all tokens for every loop step
            # for both case, next_token_logits takes the last token predicted
            # e.g., for a sequence "I talk with my", 
            #   use_cache     will generate "friend", 
            #   not_use_cache will generate "I talk with my friend"
            #   so we only need the last token "friend" as newly generated, and append to ``input_ids'', see below
            # TODO: implement

            assert 1==2  # remove this line
            print(outputs.logits.size())  # debug and remove this line
            # next_token_logits should be last token in L dimension of outputs.logits
            next_token_logits = None # MODIFY THIS

            # get argmax of the next_token_logits as the predicted next token, for greedy search
            # use torch.argmax, note the dimension should be over vocabulary size
            print(next_token_logits.size()) # remove this line
            next_tokens = None  # MODIFY THIS

            # update input_ids, by appending the new generated "next_token"
            # check dimension, input_ids is of [B,L], next_tokens should be made [B,1]
            # use torch.cat to concatenate input_ids and next_tokens to make [B, L+1]
            input_ids = None  # MODIFY THIS

            # output sentence logprob by adding current token's logprob
            # use F.log_softmax over next_token_logits, note which dimension shoud be applied ?
            next_token_logprob = None  # MODIFY THIS, check dimension (batch_size=1, vocab_size)

            # DO NOT MODIFY, the argmax token should have the max next_token_logprob
            # get the max next_token_logprob and add to likelihood
            # we accumulate batch_scores with this token's logprob as the current sentence logprob
            batch_scores += next_token_logprob.max(-1)[0]

            ###########################################
            # TODO-Explain, DO NOT MODIFY
            # where are outputs.past_key_values returned ?
            # why we reset KV_cache["past_key_values"] to
            ###########################################
            if "past_key_values" in outputs:
                KV_cache["past_key_values"] = outputs.past_key_values
            else:
                KV_cache["past_key_values"] = None

            # update attention mask
            # the attention mask is growing with the for loop from [1,...,1] -> [1,...,1,1]
            if "attention_mask" in KV_cache:
                attention_mask = KV_cache["attention_mask"]
                # TODO: Explain and DO NOT MODIFY
                # For the initial round input, if we have 4 tokens input (as prompt)
                # attention mask is starting from [1,1,1,1] -> [1,1,1,1,1] -> [1,1,1,1,1,1]......
                KV_cache["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

            # here suppose we have [B] sentences, we check if any of them has finished
            # if eos_token was found in one sentence, set that sentence to finished
            # update unfinished sequences, if 1---unfinished, if 0--finished
            if torch.any(next_tokens.eq(eos_token_id_tensor)) or input_ids.shape[-1] >= max_length:
                # print(next_tokens.eq(eos_token_id_tensor))
                break

        return input_ids, batch_scores
