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
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it", device='cuda')
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it").to('cuda')

# %%
import itertools

wmdp = pd.read_parquet("hf://datasets/cais/wmdp/wmdp-bio/test-00000-of-00001.parquet")
owt = load_dataset("Skylion007/openwebtext", streaming=True)

# Duplicate each row 3 more times
rep_wmdp = wmdp.loc[wmdp.index.repeat(4)].reset_index(drop=True)

ids = itertools.cycle(list(range(1, 5)))
rep_wmdp['perm_id'] = [next(ids) for _ in range(len(rep_wmdp))]
rep_wmdp['question_id'] = (rep_wmdp.index % 4 == 0).cumsum()

# Change correct answer index
mcq = ['A', 'B', 'C', 'D']
rep_wmdp['answer'] = (rep_wmdp['answer'] + rep_wmdp['perm_id'] - 1) % 4
rep_wmdp['mcq_answer'] = rep_wmdp['answer'].apply(lambda x: mcq[x])

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

rep_wmdp['choices'] = rep_wmdp.apply(lambda x: list(x['choices'][-(x['perm_id']-1):]) + list(x['choices'][:-(x['perm_id']-1)]), axis=1)

rep_wmdp['chat_templates'] = rep_wmdp.apply(
    lambda x: chat + [{
        "role": "user",
        "content": "The following is a multiple choice question. Please type only the letter corresponding to the answer.\nQuestion: " + x['question'] + "\n" + '\n'.join([f'{mcq[i]}. {c}' for i, c in enumerate(x['choices'])])
    }],
    axis=1
)
rep_wmdp['prompt'] = rep_wmdp['chat_templates'].apply(lambda x: tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True))
rep_wmdp['prompt_tok_len'] = rep_wmdp['prompt'].apply(lambda x: len(tokenizer(x, add_special_tokens=False)['input_ids']))
# %%
rep_wmdp = rep_wmdp[rep_wmdp['prompt_tok_len'] < 250].reset_index(drop=True)

# %%
# In batches of 50 at a time, run the model on the prompts
from transformer_lens import utils

toks = tokenizer(rep_wmdp['prompt'].tolist(), padding=True, add_special_tokens=False, return_tensors='pt')['input_ids'].to('cuda')
print(toks.shape)
model.eval()
model = torch.compile(model)
preds_list = []
N = 100
with torch.no_grad():
    for i in tqdm(range(0, len(toks), N)):
        preds = torch.softmax(model(toks[i:i+N]).logits[:, -1, :], dim=-1).argmax(dim=-1)

        # Update dataframe with predictions
        mcqa = [tokenizer.decode(p) for p in preds]
        preds_list.extend(mcqa)

for j in range(len(preds_list)):
    rep_wmdp.at[j, 'pred'] = preds_list[j]

# %%
rep_wmdp.to_csv('wmdp_bio_gemma-2-2b-it.csv', index=False)