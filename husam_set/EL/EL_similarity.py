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
from rdkit.Chem import DataStructs
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
import csv

# Setting logging low
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

# How many cores for multiprocessing
n_cores = 24
# Whether to use GPU for generating molecules with DeLinker
use_gpu = True

if not use_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load molecules
generated_smiles = frag_utils.read_triples_file("./EL.smi")

in_mols = [smi[1] for smi in generated_smiles]
frag_mols = [smi[0] for smi in generated_smiles]
gen_mols = [smi[2] for smi in generated_smiles]

du = Chem.MolFromSmiles('*')
clean_frags = [Chem.MolToSmiles(Chem.RemoveHs(AllChem.ReplaceSubstructs(Chem.MolFromSmiles(smi),du,Chem.MolFromSmiles('[H]'),True)[0])) for smi in frag_mols]

clear_output(wait=True)
print("Done")

# Check valid
results = []
for in_mol, frag_mol, gen_mol, clean_frag in zip(in_mols, frag_mols, gen_mols, clean_frags):
    if len(Chem.MolFromSmiles(gen_mol).GetSubstructMatch(Chem.MolFromSmiles(clean_frag)))>0:
        results.append([in_mol, frag_mol, gen_mol, clean_frag, gen_mol])

print("Number of generated SMILES: \t%d" % len(generated_smiles))
print("Number of valid SMILES: \t%d" % len(results))
print("%% Valid: \t\t\t%.2f%%" % (len(results)/len(generated_smiles)*100))

# Determine linkers of generated molecules
linkers = Parallel(n_jobs=n_cores)(delayed(frag_utils.get_linker)(Chem.MolFromSmiles(m[2]), Chem.MolFromSmiles(m[3]), m[1]) \
                                   for m in results)
# Standardise linkers
for i, linker in enumerate(linkers):
    if linker == "":
        continue
    linker = Chem.MolFromSmiles(re.sub('[0-9]+\*', '*', linker))
    Chem.rdmolops.RemoveStereochemistry(linker)
    linkers[i] = MolStandardize.canonicalize_tautomer_smiles(Chem.MolToSmiles(linker))
# Update results
for i in range(len(results)):
    results[i].append(linkers[i])
    
clear_output(wait=True)
print("Done")

# Create dictionary of results
results_dict = {}
for res in results:
    if res[0]+'.'+res[1] in results_dict: # Unique identifier - starting fragments and original molecule
        results_dict[res[0]+'.'+res[1]].append(tuple(res))
    else:
        results_dict[res[0]+'.'+res[1]] = [tuple(res)]

# Check proportion recovered
recovered = frag_utils.check_recovered_original_mol(list(results_dict.values()))

print("Recovered: %.2f%%" % (sum(recovered)/len(results_dict.values())*100))

# Check uniqueness
print("Unique molecules: %.2f%%" % (frag_utils.unique(results_dict.values())*100))

# Check if molecules pass 2D filters 
filters_2d = frag_utils.calc_filters_2d_dataset(results, pains_smarts_loc="./wehi_pains.csv", n_cores=n_cores)

results_filt = []
for res, filt in zip(results, filters_2d):
    if filt[0] and filt[1] and filt[2]:
        results_filt.append(res)
        
clear_output(wait=True)        
print("Pass all 2D filters: \t\t\t\t%.2f%%" % (len(results_filt)/len(results)*100))
print("Valid and pass all 2D filters: \t\t\t%.2f%%" % (len(results_filt)/len(generated_smiles)*100))
print("Pass synthetic accessibility (SA) filter: \t%.2f%%" % (len([f for f in filters_2d if f[0]])/len(filters_2d)*100))
print("Pass ring aromaticity filter: \t\t\t%.2f%%" % (len([f for f in filters_2d if f[1]])/len(filters_2d)*100))
print("Pass SA and ring filters: \t\t\t%.2f%%" % (len([f for f in filters_2d if f[0] and f[1]])/len(filters_2d)*100))
print("Pass PAINS filters: \t\t\t\t%.2f%%" % (len([f for f in filters_2d if f[2]])/len(filters_2d)*100))

# Get unique molecules
print("Number molecules passing 2D filters:\t\t%d" % len(results_filt))
results_filt_unique = example_utils.unique_mols(results_filt)
print("Number unique molecules passing 2D filters:\t%d" % len(results_filt_unique))

# Calculate Tanimoto similarity of Morgan fingerprints (radius 2, 2048 bits)
smis = [res[2] for res in results_filt_unique]
gen_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, 2048) for smi in smis]
orig_fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(in_mols[0]), 2, 2048)

gen_sims = [DataStructs.TanimotoSimilarity(orig_fp, gen_fp) for gen_fp in gen_fps]

best_mols = sorted(list(zip(smis, gen_sims)), key=lambda x: x[1], reverse=True)[:10]

mols = [Chem.MolFromSmiles(m[0]) for m in best_mols]

# get chEMBL IDs
similarity = new_client.similarity
best_mols_sim = [(m[0]) for m in best_mols]

res = []
chem = []
for similar_mols in best_mols_sim:
    res_s = similarity.filter(smiles=similar_mols, similarity=70).only(['molecule_chembl_id', 'similarity'])
    if res_s[0] is not None:
        chem.append(res_s[0]['molecule_chembl_id'])
        res.append("%.2f" % float(res_s[0]['similarity']))
    else:
        res.append('-')
        chem.append('-')

# tuple for the legends
legend_tup = zip(["%.2f" % m[1] for m in best_mols], chem, res)
legend_tup_list = []
for m in legend_tup:
    legend_tup_list.append(m[0] + ", " + m[1] + ", " + m[2])

img_similar = Draw.MolsToGridImage(mols,
                     subImgSize=(500, 500), molsPerRow=5, legends=[m for m in legend_tup_list],
                    )
with open('grid1.svg', 'w') as f:
    f.write(img_similar.data)
cairosvg.svg2png( url='grid1.svg', write_to= './images/best_rmsd_develop.png' )
os.remove('grid1.svg')
print('The best_rmsd_develop.png is saved in specified folder!')