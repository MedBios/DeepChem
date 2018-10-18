
# coding: utf-8

# In[ ]:


import os
import numpy as np
import random


# In[ ]:


base_dir = "/media/external/pubchem_compound_3d/smiles/"


# In[ ]:


files = os.listdir(base_dir)
csv_range = int(files[0].split('.')[0].split('-')[1]) - int(files[0].split('.')[0].split('-')[0]) + 1
fmap = dict((int(x.split('.')[0].split('-')[0]), x) for x in files)


# In[ ]:


def get_smiles(cid):
    start_idx = (cid // csv_range) * csv_range + 1
    with open(base_dir + fmap[start_idx], "r") as f:
        for i in range(cid - start_idx + 1):
            next(f)
        _, c_smiles, _ = f.readline().strip().split(',')
    return c_smiles[1:-1]


# In[ ]:


def text2image(text):
    image = np.zeros((8, len(text)), dtype=np.uint8)
    for index, character in enumerate(text):
        y = list(bin(ord(character)))
        if len(y) <= 10:
            for j in range(2, min(8, len(y))):
                image[j - 2, index] = int(y[len(y) + 1 - j])
    return image


# In[ ]:


def text2image_batch(strings):
    image = np.zeros((len(strings), 8, len(strings[0])), dtype=np.uint8)
    for text_idx, text in enumerate(strings):
        for index, character in enumerate(text):
            y = list(bin(ord(character)))
            if len(y) <= 10:
                for j in range(2, min(8, len(y))):
                    image[text_idx, j - 2, index] = int(y[len(y) + 1 - j])
    return image


# In[3]:


def pad_to_max(strings, max_len=None):
    if max_len == None:
        max_len = max(map(len, strings))
    return [x.ljust(max_len, chr(0))[:max_len] for x in strings]


# In[ ]:


def get_max_cid():
    return int(open(base_dir + fmap[max(fmap.keys())], 'r').readlines()[-1].split(',')[0])


# In[ ]:


def get_random_cids(n, exclude_list=[]):
    max_cid = get_max_cid()
    cids = []
    for i in range(n):
        cids.append(random.randint(1, max_cid))
        while cids[-1] in exclude_list:
            cids[-1] = random.randint(1, max_cid)
    return cids


# In[ ]:


if __name__ == "__main__":
    cids = get_random_cids(3000, exclude_list=list(range(1, 100001)))
    print(cids)


# In[ ]:


if __name__ == "__main__":
    smiles = list(map(get_smiles, range(1, 101)))
    smiles = pad_to_max(smiles)
    print(text2image_batch(smiles).shape)


# In[ ]:


if __name__ == "__main__":
    for i in range(1 + 55047830, 1000001 + 55047830, 1000):
        smiles = list(map(get_smiles, range(i, i + 1000)))
        print(i, len(smiles))


# In[2]:


if __name__ == "__main__":
    get_ipython().system('jupyter nbconvert --to=python retrieve_pubchem_compound.ipynb')

