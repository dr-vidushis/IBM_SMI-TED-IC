# Predicting IONIC properties with fine-tuned SMI.TED-289M 

# %%
# System
import sys


# Machine learning
import torch

# Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#metrics
from sklearn.metrics import mean_squared_error

# Chemistry
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import Descriptors
PandasTools.RenderImagesInAllDataFrames(True)

# materials.smi-ted (smi-ted)
from smi_ted_light.load import load_smi_ted
import fast_transformers

def normalize_smiles(smi, canonical, isomeric):
    try:
        normalized = Chem.MolToSmiles(
        Chem.MolFromSmiles(smi), canonical=canonical, isomericSmiles=isomeric
        )
    except:
        normalized = None
    return normalized

# %%
def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# %% [markdown]
# ## Load Foundation Models

# %%
model_smi_ted = load_smi_ted( 
    folder='smi_ted_light',
    ckpt_filename='smi-ted-Light-Finetune_epoch=479_ionic_seed90_rmse=0.1087.pt',
)
model_smi_ted.eval()

# %% [markdown]
# ## Load datasets

# %%
## import data for prediction
df_ionic_test = pd.read_csv('data/HIL.csv')

# %%
with torch.no_grad():
    df_test_emb = model_smi_ted.encode(df_ionic_test['smiles'])

# %%
df_test_emb['temperature'] = df_ionic_test['temperature']

# %%
torch_emb = torch.tensor(df_test_emb.values)

# %%
outputs = model_smi_ted.net(torch_emb.float()).cpu().detach().numpy()

# %%
df_ionic_test['ionic_predicted'] = outputs

# %%

df_ionic_test.to_csv(f'HIL_predictions.csv')

# %%

