# %% [markdown]
# ## Goal: Replicate the "Applying Sparse Autoencoders to Unlearn Knowledge in Language Models" paper
# 
# Link: https://arxiv.org/pdf/2410.19278

# %%
import os
import torch

from datasets import load_dataset
from sae_lens import SAE
from transformer_lens import HookedTransformer
import pandas as pd

from tqdm import tqdm
os.environ["HF_TOKEN"] = "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"

# %%
model = HookedTransformer.from_pretrained(
    'google/gemma-2-2b-it',
    default_padding_side='left'
)

tokenizer = model.tokenizer


# %%
rep_wmdp = pd.read_csv('wmdp_bio_gemma-2-2b-it.csv')

# %%
correct = rep_wmdp.groupby('question_id').apply(lambda x: (x['mcq_answer'] == x['pred'].str.strip()).sum() >= 4)

# %%
# Get rows of rep_wmdp that are correct
rep_wmdp = rep_wmdp[rep_wmdp['question_id'].isin(correct[correct].index)]
# Only keep first of each question_id
rep_wmdp = rep_wmdp.groupby('question_id').first().reset_index()
# %%
import einops

mmlu = load_dataset("cais/mmlu", "all", streaming=True)

mmlu = mmlu['auxiliary_train'].shuffle(seed=42, buffer_size=len(rep_wmdp)).take(len(rep_wmdp))

mcq_mmlu = {
    'question': [q['question'] for q in iter(mmlu)],
    'choices': [q['choices'] for q in iter(mmlu)],
    'answer': [q['answer'] for q in iter(mmlu)]
}
mcq_mmlu = pd.DataFrame(mcq_mmlu)

# %%
mcq = ['A', 'B', 'C', 'D']
mcq_mmlu['mcq_answer'] = mcq_mmlu['answer'].apply(lambda x: mcq[x])

# %%
few_shot_q = """The following is a multiple choice question. Please type only the letter corresponding to the answer.
Question: What is the largest organ?
A. Liver
B. Skin
C. Heart
D. Brain
"""
few_shot_a = "B"

chat = [
    { "role": "user", "content": few_shot_q},
    { "role": "model", "content": few_shot_a}
]

mcq_mmlu['chat_templates'] = mcq_mmlu.apply(
    lambda x: chat + [{
        "role": "user",
        "content": "The following is a multiple choice question. Please type only the letter corresponding to the answer.\nQuestion: " + x['question'] + "\n" + '\n'.join([f'{mcq[i]}. {c}' for i, c in enumerate(x['choices'])])
    }],
    axis=1
)
mcq_mmlu['prompt'] = mcq_mmlu['chat_templates'].apply(lambda x: tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True))
mcq_mmlu['prompt_tok_len'] = mcq_mmlu['prompt'].apply(lambda x: len(tokenizer(x, add_special_tokens=False)['input_ids']))

# %%
from matplotlib import pyplot as plt

plt.hist(mcq_mmlu['prompt_tok_len'], bins=100)
plt.show()

# %%
mcq_mmlu = mcq_mmlu[mcq_mmlu['prompt_tok_len'] < 600].reset_index(drop=True)

# %%
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "gemma-scope-2b-pt-mlp", 
    sae_id = "layer_9/width_16k/average_l0_38", 
    device='cuda'
)

# %%
import gc

wmdp_toks = tokenizer(rep_wmdp['prompt'].tolist(), padding=True, add_special_tokens=False, return_tensors='pt')['input_ids']
mmlu_toks = tokenizer(mcq_mmlu['prompt'].tolist(), padding=True, add_special_tokens=False, return_tensors='pt')['input_ids']
model.eval()
rep_wmdp['pred'] = None
N = 30

wmdp_feat_act_hist = torch.zeros(sae.cfg.d_sae, device='cuda')

torch.cuda.empty_cache()
gc.collect()

with torch.no_grad():
    for i in tqdm(range(0, len(wmdp_toks), N)):
        _, wmdp_cache = model.run_with_cache(
            wmdp_toks[i:i+N],
            names_filter=[sae.cfg.hook_name]
        )
        sae_feats = sae.encode(wmdp_cache[sae.cfg.hook_name])
        del wmdp_cache
        torch.cuda.empty_cache()
        gc.collect()
        sae_feats[sae_feats > 0] = 1
        sae_feats[sae_feats < 0] = 0
        wmdp_feat_act_hist += sae_feats.sum(dim=1).sum(dim=0)

N = 10
mmlu_feat_act_hist = torch.zeros(sae.cfg.d_sae, device='cuda')
with torch.no_grad():
    for i in tqdm(range(0, len(mmlu_toks), N)):
        _, mmlu_cache = model.run_with_cache(
            mmlu_toks[i:i+N],
            names_filter=[sae.cfg.hook_name]
        )
        sae_feats = sae.encode(mmlu_cache[sae.cfg.hook_name])
        del mmlu_cache
        torch.cuda.empty_cache()
        gc.collect()
        sae_feats[sae_feats > 0] = 1
        sae_feats[sae_feats < 0] = 0
        mmlu_feat_act_hist += sae_feats.sum(dim=1).sum(dim=0)

# %%
# Save the feat act hists

torch.save(wmdp_feat_act_hist, 'wmdp_feat_act_hist.pt')
torch.save(mmlu_feat_act_hist, 'mmlu_feat_act_hist.pt')

# %%
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(wmdp_feat_act_hist.cpu().numpy() / rep_wmdp['prompt_tok_len'].sum(), mmlu_feat_act_hist.cpu().numpy() / mcq_mmlu['prompt_tok_len'].sum())
plt.xlabel("WMDP")
plt.ylabel("Wiki")
plt.show()

# %%


# %%



