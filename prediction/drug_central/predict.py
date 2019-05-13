
# coding: utf-8

# In[15]:


import pandas as pd
import sys
sys.path.append("../../../ddt/")
from utility import FeatureGenerator
from sklearn.externals import joblib
import os
MODELS_DIR = "../../models"
PY_VERSION = '27' if sys.version_info[0] < 3 else '35'
import multiprocessing as mp
import json
import warnings
warnings.filterwarnings("ignore")


# In[16]:


samples_df = pd.read_csv("sellect_drug_library.csv")
# samples_df.head()


# In[14]:


def get_models():    
    model_names = open('../../py' + PY_VERSION + '_models.txt', 'r').readlines()[0].split(',')
    for model in model_names:
        yield model, joblib.load(os.path.join(MODELS_DIR, model))


# In[8]:


def get_prediction(smiles, confidence=0.9):
    actives = []
    ft = FeatureGenerator()
    ft.load_smiles(smiles)
    features = ft.extract_tpatf()
    for model_name, model in get_models():
        if type(model).__name__ == "SVC":
            pred = model.predict(features)
            if pred[0] == 1.0: actives.append(model_name)
        else:
            pred = model.predict_proba(features)
            if pred[0][1] > confidence: actives.append(model_name)
    return smiles, actives


# In[10]:


pool = mp.Pool(mp.cpu_count())
results = [pool.apply_async(get_prediction, args=(smiles,)) for smiles in samples_df.SMILES.tolist()]
output = [p.get() for p in results]


# In[24]:


output_dict = {}
for _smiles, _actives in output:
    output_dict[_smiles] =_actives
json.dump(output_dict, open('py' + PY_VERSION + '_results.json', 'w'))

