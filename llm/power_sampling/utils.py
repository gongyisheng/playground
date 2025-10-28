import numpy as np
import random
import torch
from torch.nn import functional as F
import transformers
from tqdm import tqdm


def format_prompt(question: str) -> str:
    output_instruct = "\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    return question + output_instruct

def naive_samp(
        model, 
        tokenizer, 
        input_ids, 
        temp: float, 
        max_new_tokens: int,
    ):
    input_len = len(input_ids[0])
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens, 
        do_sample = True,
        temperature = temp,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True, 
        output_scores=True, # only works when set return_dict_in_generate = True
        output_logits=True, # only works when set return_dict_in_generate = True
    )

    unscaled_logits = torch.stack(output.logits, dim=0) # [seq_len]
    scaled_logits = torch.stack(output.scores, dim=0)   # [seq_len]
    tokens = output.sequences[0][input_len:]            # [seq_len-context]
    
    idx = tokens.view(unscaled_logits.shape[0], 1, 1)
    log_probs_unnorm = (1/temp * torch.gather(F.log_softmax(unscaled_logits), -1, idx)).view(-1).tolist()
    log_probs_norm = torch.gather(F.log_softmax(scaled_logits), -1, idx).view(-1).tolist()

    return output, log_probs_norm, log_probs_unnorm

def mcmc_power_samp(
        model, 
        tokenizer, 
        input_ids,
        temp: float, 
        mcmc_steps: int, 
        max_new_tokens: int, 
        block_num: int = 16
    ):

    input_len = len(input_ids[0])
    sequence_round = input_ids

    # print(f'max_new_tokens:{max_new_tokens}')
    assert max_new_tokens % block_num == 0
    jump_size = int(max_new_tokens // block_num)
    # print(f'jump_size:{jump_size}')
    attempts = 0
    acceptances = 0

    for _ in range(block_num):
        sequence_base_len = len(sequence_round[0])
        output_round, lp_norm_round, lp_unnorm_round = naive_samp(model, tokenizer, sequence_round, temp, jump_size)
        sequence_round = output_round.sequences
        
        for _ in tqdm(range(mcmc_steps)):
            attempts += 1
            sequence_round_len = len(sequence_round[0])
            idx_abs = random.randint(sequence_base_len, sequence_round_len-1)
            idx_rel = idx_abs - sequence_base_len

            output_prop, lp_norm_prop, lp_unnorm_prop = naive_samp(model, tokenizer, sequence_round[:,:idx_abs], temp, sequence_round_len-idx_abs)
            
            lp_norm_round = lp_norm_round.copy()[idx_rel:] # log_prob_cur
            lp_unnorm_round = lp_norm_round.copy()[idx_rel:] # target_log_prob_cur
            log_r = sum(lp_unnorm_prop) + sum(lp_norm_round) - sum(lp_unnorm_round) - sum(lp_norm_prop)

            if np.random.rand() < np.exp(log_r):
                acceptances+=1
                sequence_round = output_prop.sequences
                lp_norm_round = lp_norm_prop.copy()
                lp_unnorm_round = lp_unnorm_prop.copy()
        
        output_ids_list = output_round.sequences[0].tolist()
        if tokenizer.eos_token_id in output_ids_list:
            eos_idx = output_ids_list.index(tokenizer.eos_token_id)
            output_ids_list = output_ids_list[:eos_idx + 1]
            acceptance_ratio = acceptances/attempts
            return output_ids_list, acceptance_ratio

    output_ids_list = output_round.sequences[0].tolist()
    acceptance_ratio = acceptances/attempts
    return output_ids_list, acceptance_ratio