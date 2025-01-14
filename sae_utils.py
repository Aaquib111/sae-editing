import torch
from collections import defaultdict

def get_batch_sae_acts(model, tokenizer, prompt_list, subject_positions, sae_layers, sae_dict, concatenate_batch_together=True):
    if not isinstance(prompt_list, list):
        prompt_list = [prompt_list]

    tokenized_batch = tokenizer(prompt_list, return_tensors="pt", padding=True)
    output = model(tokenized_batch["input_ids"].cuda(), attention_mask=tokenized_batch["attention_mask"].cuda(), output_hidden_states=True)
    if concatenate_batch_together:
        acts = {}
        sae_acts = {}
        for layer in sae_layers:
            # num_subject_tokens = 0
            total_hidden_states = output.hidden_states[layer+1]
            answer_hidden_states = []
            for i in range(len(total_hidden_states)):
                answer_hidden_states.append(total_hidden_states[i][subject_positions[i]])
            acts[layer] = torch.cat(answer_hidden_states, dim=0)
            sae_acts[layer] = sae_dict[layer].encode(acts[layer])
    else:
        acts = defaultdict(list)
        sae_acts = defaultdict(list)
        for layer in sae_layers:
            total_hidden_states = output.hidden_states[layer+1]
            answer_hidden_states = []
            for i in range(len(total_hidden_states)):
                prompt_acts = total_hidden_states[i][subject_positions[i]]
                acts[layer].append(prompt_acts)
                sae_acts[layer].append(sae_dict[layer].encode(prompt_acts))
    return sae_acts


def get_feature_indices(sae_acts):
    """
    sae_acts should be some dict of sae_acts, key is layer, value is a tensor or list of [..., n_features]
    """
    for layer in sae_acts:
        if isinstance(sae_acts[layer], list):
            for i in range(len(sae_acts[layer])):
                print(sae_acts[layer][i].shape)
        else:
            assert isinstance(sae_acts[layer], torch.Tensor), "values of sae_acts should be a list of tensors or a tensor"
            print(sae_acts[layer].shape)

import pandas as pd
def load_explanation_dicts(model_name, num_layers):
    explanation_dfs = {}
    for layer in range(num_layers):
        explanation_df = pd.read_csv(f"saes/explanation_dfs/{model_name}/{layer}.csv")
        explanation_dfs[layer] = explanation_df
    return explanation_dfs