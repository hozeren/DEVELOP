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
from EL_2D import results_filt_unique, results_filt


# Setting logging low
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

#Load Fragments
PATH = os.getcwd()
scaff_1_path = '././pyridine_celebrex.sdf'

# How many cores for multiprocessing
n_cores = 4
# Whether to use GPU for generating molecules with DeLinker
use_gpu = True

scaff_1_sdf = Chem.SDMolSupplier(scaff_1_path)
scaff_1_smi = Chem.MolToSmiles(scaff_1_sdf[0])

if not use_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Generate conformers
_ = rdkit_conf_parallel.gen_confs([res[2] for res in results_filt_unique], "./DeLinker_generated_mols_unique.sdf",
                                  smi_frags=[res[1] for res in results_filt_unique], numcores=n_cores, jpsettings=True)
clear_output(wait=True)
print("Done")

# Load conformers
gen_sdfs = Chem.SDMolSupplier("./DeLinker_generated_mols_unique.sdf")
ref_mol = Chem.Mol(scaff_1_sdf[0])

# Get list of starting fragments and original molecules
used = set([])
ref_identifiers = [(res[1], res[0]) for res in results_filt if res[1]+'.'+res[0] not in used and (used.add(res[1]+'.'+res[0]) or True)]

# Get indices of compounds in SD file
start_stop_idxs = []
start = 0
errors = 0

curr_st_pt = ""
for count, gen_mol in enumerate(gen_sdfs):
    try:
        # Check if seen this ligand before
        if gen_mol.GetProp("_Model") == str(0):
            stop = count
            if count != 0:
                start_stop_idxs.append((start, stop))
            start = int(stop) # deep copy
            curr_st_pt = gen_mol.GetProp("_StartingPoint")
    except:
        errors += 1
        continue

# Add last
start_stop_idxs.append((start, len(gen_sdfs)))

# Calculate SC_RDKit fragments scores
names_frags = []
best_scores_frags = []
idx_best_poses_frags = []
names_frags_start_pts = []

with Parallel(n_jobs=n_cores, backend='multiprocessing') as parallel:
    for i in range(-(-len(start_stop_idxs)//n_cores)):
        jobs = []
        for core in range(n_cores):
            if i*n_cores+core < len(start_stop_idxs):
                start, stop = start_stop_idxs[i*n_cores+core]
                frag_smi = gen_sdfs[start].GetProp("_StartingPoint")
                # Prepare jobs
                gen_mols = [(Chem.Mol(gen_sdfs[idx]), Chem.Mol(ref_mol), str(frag_smi)) for idx in range(start, stop) if gen_sdfs[idx] is not None] # Test addition
                jobs.append(gen_mols)

        # Get SC_RDKit scores
        set_scores = parallel((delayed(frag_utils.SC_RDKit_frag_scores)(gen_mols) for gen_mols in jobs))
        for core, scores in enumerate(set_scores):
            start, stop = start_stop_idxs[i*n_cores+core]
            names_frags.append(gen_sdfs[start].GetProp("_Name"))
            names_frags_start_pts.append(gen_sdfs[start].GetProp("_StartingPoint"))
            best_scores_frags.append(max(scores))
            idx_best_poses_frags.append((np.argmax(scores)+start, 0))
            
best_scores_frags_all = []
comp = list(zip(names_frags_start_pts, names_frags))
for res in results_filt:
    try:
        idx = comp.index((res[1], res[2]))
        best_scores_frags_all.append(best_scores_frags[idx])
    except:
        continue

# Print SC_RDKit Fragments results
print("Average SC_RDKit Fragments score: %.3f +- %.3f\n" % (np.mean(best_scores_frags_all), np.std(best_scores_frags_all)))

thresholds_SC_RDKit = [0.6, 0.7, 0.75, 0.8]
for thresh in thresholds_SC_RDKit:
    print("SC_RDKit Fragments - Molecules above %.2f: %.2f%%" % (thresh, len([score for score in best_scores_frags_all if score >= thresh]) / len(best_scores_frags_all)*100))

# Calculate fragments RMSDs
names_rmsd_frags = []
best_rmsd_frags = []
idx_best_rmsd_poses_frags = []
names_rmsd_frags_start_pts = []

with Parallel(n_jobs=n_cores, backend='multiprocessing') as parallel:
    for i in range(-(-len(start_stop_idxs)//n_cores)):
        jobs = []
        for core in range(n_cores):
            if i*n_cores+core < len(start_stop_idxs):
                start, stop = start_stop_idxs[i*n_cores+core]
                frag_smi = gen_sdfs[start].GetProp("_StartingPoint")
                # Prepare jobs
                gen_mols = [(Chem.Mol(gen_sdfs[idx]), Chem.Mol(ref_mol), str(frag_smi)) for idx in range(start, stop) if gen_sdfs[idx] is not None] # Test addition
                jobs.append(gen_mols)

        # Calculate RMSDs
        set_scores = parallel((delayed(frag_utils.rmsd_frag_scores)(gen_mols) for gen_mols in jobs)) # Multiprocessing step
        for core, scores in enumerate(set_scores):
            start, stop = start_stop_idxs[i*n_cores+core]
            names_rmsd_frags.append(gen_sdfs[start].GetProp("_Name"))
            names_rmsd_frags_start_pts.append(gen_sdfs[start].GetProp("_StartingPoint"))
            best_rmsd_frags.append(min(scores))
            idx_best_rmsd_poses_frags.append((np.argmin(scores)+start, 0))
            
best_rmsd_frags_all = []
comp = list(zip(names_rmsd_frags_start_pts, names_rmsd_frags))
for res in results_filt:
    try:
        idx = comp.index((res[1], res[2]))
        best_rmsd_frags_all.append(best_rmsd_frags[idx])
    except:
        continue

# Print RMSD Fragments results
print("Average Fragments RMSD: %.3f +- %.3f\n" % (np.mean(best_rmsd_frags_all), np.std(best_rmsd_frags_all)))

thresholds_rmsd = [1.0, 0.75, 0.5]
for thresh in thresholds_rmsd:
        print("RMSD Fragments - Molecules below %.2f: %.2f%%" % (thresh, len([score for score in best_rmsd_frags_all if score <= thresh]) / len(best_rmsd_frags_all)*100))

