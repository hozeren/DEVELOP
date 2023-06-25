#!/usr/bin/env python

import sys
from unittest import result
from matplotlib.pyplot import grid

sys.path.append("/home/kevin/Desktop/DeLinker/examples")

# Autorship information
__author__ = "Hüsamettin Deniz Özeren"
__copyright__ = "Copyright 2021"
__credits__ = ["Hüsamettin Deniz Özeren"]
__license__ = "GNU General Public License v3.0"
__maintainer__ = "Hüsamettin Deniz Özeren"
__email__ = "denizozeren614@gmail.com"

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import MolDrawing, DrawingOptions
from rdkit.Chem import MolStandardize
from rdkit.Chem.Draw import rdMolDraw2D
from chembl_webresource_client.new_client import new_client

import numpy as np
import cairosvg
import os

from itertools import product
from joblib import Parallel, delayed
import re
from collections import defaultdict

from IPython.display import clear_output
IPythonConsole.ipython_useSVG = True

import DEVELOP
from DEVELOP import DenseGGNNChemModel
import frag_utils
import rdkit_conf_parallel
import example_utils
from EL_start import fragments
from EL_3D import names_frags, best_scores_frags, best_rmsd_frags, names_rmsd_frags


# Setting logging low
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


#Load Fragments
PATH = os.getcwd()
scaff_1_path = './pyridine_celebrex.sdf'

# How many cores for multiprocessing
n_cores = 24
# Whether to use GPU for generating molecules with DeLinker
use_gpu = True

scaff_1_sdf = Chem.SDMolSupplier(scaff_1_path)
scaff_1_smi = Chem.MolToSmiles(scaff_1_sdf[0])

if not use_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Best by SC_RDKit Fragments
best_mols = sorted(list(zip(names_frags, best_scores_frags)), key=lambda x: x[1], reverse=True)[:10]

mols = [Chem.MolFromSmiles(m[0]) for m in best_mols]
frag_to_align = re.sub('\[\*\]', '', fragments.split('.')[0])
p = Chem.MolFromSmiles(frag_to_align)
AllChem.Compute2DCoords(p)

for m in mols: AllChem.GenerateDepictionMatching2DStructure(m,p)
img = Draw.MolsToGridImage(mols,
                     subImgSize=(500, 500), molsPerRow=5, legends=["%.2f" % m[1] for m in best_mols]
                    )
with open('grid1.svg', 'w') as f:
    f.write(img.data)
cairosvg.svg2png( url='grid1.svg', write_to= './images/best_fragments.png' )
os.remove('grid1.svg')

# Best by RMSD Fragments
best_mols = sorted(list(zip(names_rmsd_frags, best_rmsd_frags)), key=lambda x: x[1])[:10]

mols = [Chem.MolFromSmiles(m[0]) for m in best_mols]
frag_to_align = re.sub('\[\*\]', '', fragments.split('.')[0])
p = Chem.MolFromSmiles(frag_to_align)
AllChem.Compute2DCoords(p)

# get chEMBL IDs
similarity = new_client.similarity
best_mols_sim = [(m[0]) for m in best_mols]
res = []
chem = []
for similar_mols in best_mols_sim:
    res_s = similarity.filter(smiles=similar_mols, similarity=70).only(['molecule_chembl_id', 'similarity'])
    if res_s[0] is not None:
        chem.append(res_s[0]['molecule_chembl_id'])
        res.append("%.2f" % res_s[0]['similarity'])
    else:
        res.append('-')
        chem.append('-')

# tuple for the legends
legend_tup = zip(["%.2f" % m[1] for m in best_mols], chem, res)
legend_tup_list = []
for m in legend_tup:
    legend_tup_list.append(m[0] + ", " + m[1] + ", " + m[2])

for m in mols: AllChem.GenerateDepictionMatching2DStructure(m,p)
img2 = Draw.MolsToGridImage(mols,
                     subImgSize=(500, 500), molsPerRow=5, legends=[m for m in legend_tup_list]
                    )
with open('grid1.svg', 'w') as f:
    f.write(img2.data)
cairosvg.svg2png( url='grid1.svg', write_to= './images/best_rmsd.png' )
os.remove('grid1.svg')
