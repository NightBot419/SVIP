# -*- coding: utf-8 -*-
import os,sys
pwd = os.getcwd()
sys.path.insert(0,pwd)
#%%
print('-'*30)
print(os.getcwd())
print('-'*30)
#%%
import pandas as pd
import numpy as np
import gensim.downloader as api
import pickle
from pathlib import Path

# Ensure the output directory exists
output_dir = Path('./attribute/w2v/')
output_dir.mkdir(parents=True, exist_ok=True)

print('Loading pre-trained w2v model (this may take a while on first run)...')
model_name = 'word2vec-google-news-300'
model = api.load(model_name)
dim_w2v = 300
print('Done loading model.')
#%%
# No words to replace for this custom attribute list
replace_word = []
#%%
path = './data/Fish/attributes.txt'
df=pd.read_csv(path,sep=' ',header = None, names = ['idx','des'])
des = df['des'].values
#%% Filter and clean descriptions
print('Cleaning attribute descriptions...')
new_des = [' '.join(i.split('_')) for i in des]
new_des = [' '.join(i.split('-')) for i in new_des]
new_des = [' '.join(i.split('::')) for i in new_des]
new_des = [i.split('(')[0] for i in new_des]
# The following line is removed as it's specific to CUB's "has_" prefix
# new_des = [i[4:] for i in new_des]
#%%
df['new_des']=new_des
# This file is not strictly necessary but good for debugging
df.to_csv('attribute/CUB/fish_new_des.csv')
print('Done preprocessing attribute descriptions.')
#%%
print('Generating word vectors for each attribute...')
all_w2v = []
for s in new_des:
    print(f"Processing: {s}")
    words = s.split(' ')
    if words[-1] == '':     #remove empty element
        words = words[:-1]
    w2v = np.zeros(dim_w2v)
    word_count = 0
    for w in words:
        if w in model:
            w2v += model[w]
            word_count += 1
        else:
            print(f"  - Warning: word '{w}' not in vocabulary.")
    # Average the vectors
    if word_count > 0:
        w2v = w2v / word_count
    all_w2v.append(w2v[np.newaxis,:])
#%%
all_w2v_np = np.concatenate(all_w2v,axis=0)
output_path = output_dir / 'Fish_attribute.pkl'
print(f"Saving attribute vectors to {output_path}")
with open(output_path, 'wb') as f:
    pickle.dump(all_w2v_np,f)

print("Finished creating fish attribute vectors.")

