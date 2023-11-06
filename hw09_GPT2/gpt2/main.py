import torch
import torch.nn as nn
import argparse
import os
import numpy as np
from time import time, sleep

import torch

from transformers import pipeline, set_seed
from transformers import GPT2Tokenizer, AutoModelForCausalLM, GPT2LMHeadModel
from modeling_my_gpt2 import GPT2LMHeadModel as MyGPT2LMHeadModel


######################
###  main file  ######
######################


# do not modify the main logic parts
# just modify some variables if needed, such as use_cuda, pretrained_model_path, or max_len
def main():
    # if you have a GPU, turn this on
    use_cuda = False  # args.cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print('The device is', device)

    set_seed(123)

    # download https://huggingface.co/gpt2/tree/main/pytorch_model.bin and put in gpt2_model/
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path="gpt2_model/", local_files_only=True)
    model = MyGPT2LMHeadModel.from_pretrained("gpt2_model/", pad_token_id=tokenizer.eos_token_id,
                                              local_files_only=True).to(device)

    model_inputs = tokenizer('I enjoy talking with', return_tensors='pt')
    model_inputs['input_ids'] = model_inputs['input_ids'].to(device)
    model_inputs['attention_mask'] = model_inputs['attention_mask'].to(device)
    print("Input attention mask", model_inputs['attention_mask'])
    print("This prompt is given by user, so can be fully attended, it's all ones\n")

    # check the last (50256) "<|endoftext|>" entry of gpt2_model/vocab.json
    eos_token_id = [50256]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(device)

    # we set max_len so that the max length of generated sentence does not exceed
    max_len = 100

    # From the colab demo, you could find that this is OpenAI official way of generating responses from prompts
    # try OpenAI API for greedy_search
    # check https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py
    print('Running Built-in generation')
    output_tensors_0 = model.generate(**model_inputs, max_new_tokens=max_len, use_cache=True, output_scores=True)
    ########################################
    ## we simplify the generation process
    ########################################

    # first check use cache option
    print('Running Use Cache')
    nl = 2
    t1 = time()
    for _ in range(nl):
        output_tensors_1, scores_1 = model.generate_greedy_search(model_inputs['input_ids'],
                                                                  model_inputs['attention_mask'], eos_token_id_tensor,
                                                                  max_length=max_len, use_cache=True)
    t2 = time()

    sleep(1)

    # then check not use cache option
    print('Running No Use Cache')
    t3 = time()
    for _ in range(nl):
        output_tensors_2, scores_2 = model.generate_greedy_search(model_inputs['input_ids'],
                                                                  model_inputs['attention_mask'], eos_token_id_tensor,
                                                                  max_length=max_len, use_cache=False)
    t4 = time()

    print("Use Official generate Output:\n" + 100 * '-')
    print(tokenizer.decode(output_tensors_0[0], skip_special_tokens=True))
    print()
    print("Our Use cache Output:\n" + 100 * '-', 'logprob:', scores_1)
    print(tokenizer.decode(output_tensors_1[0], skip_special_tokens=True))
    print()
    print("Our Not Use cache Output (should be same):\n" + 100 * '-', 'logprob:', scores_2)
    print(tokenizer.decode(output_tensors_2[0], skip_special_tokens=True))
    print()

    print()
    print('Use KV Cache time:', np.round(t2 - t1, 2))
    print('NOT USE KV Cache time:', np.round(t4 - t3, 2))


if __name__ == '__main__':
    main()
