#!/usr/bin/env python

import sys
from matplotlib.pyplot import grid

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
import data.prepare_data_linker_design
from data.prepare_data_linker_design import read_file, preprocess

# Setting logging low
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

#Load Fragments
PATH = os.getcwd()
scaff_1_path = './pyridine_celebrex.sdf'

# How many cores for multiprocessing
n_cores = 4
# Whether to use GPU for generating molecules with DeLinker
use_gpu = True

scaff_1_sdf = Chem.SDMolSupplier(scaff_1_path)
scaff_1_smi = Chem.MolToSmiles(scaff_1_sdf[0])


# This is drawing the image, needs Jupyter but it also saves as PNG
img = Draw.MolsToGridImage([Chem.MolFromSmiles(scaff_1_smi)], molsPerRow=2, subImgSize=(300, 300))
with open('grid1.svg', 'w') as f:
    f.write(img.data)

# create the path and save
if not os.path.exists(PATH+"/images"):
    images_path = os.path.join(PATH, "images")
    os.mkdir(images_path)
cairosvg.svg2png( url='grid1.svg', write_to= './images/scaffold.png')
os.remove('grid1.svg')

starting_point_2d = Chem.Mol(scaff_1_sdf[0])
_ = AllChem.Compute2DCoords(starting_point_2d)
img2 = example_utils.mol_with_atom_index(starting_point_2d)
d = rdMolDraw2D.MolDraw2DCairo(500, 500) # or MolDraw2DSVG to get SVGs
d.drawOptions().addStereoAnnotation = True
d.drawOptions().addAtomIndices = True
d.DrawMolecule(img2)
d.FinishDrawing()
d.WriteDrawingText('./images/atom_annotation_1.png') 

#selecting starting pairs
atom_pair_idx_1 = [1, 6]
atom_pair_idx_2 = [7, 8]
bonds_to_break = [starting_point_2d.GetBondBetweenAtoms(x,y).GetIdx() for x,y in [atom_pair_idx_1, atom_pair_idx_2]]

fragmented_mol = Chem.FragmentOnBonds(starting_point_2d, bonds_to_break)
_ = AllChem.Compute2DCoords(fragmented_mol)
Chem.SanitizeMol(fragmented_mol)
d2 = rdMolDraw2D.MolDraw2DCairo(500, 500) # or MolDraw2DSVG to get SVGs
fragmented_mol #for jupyter
d2.DrawMolecule(fragmented_mol)
d2.WriteDrawingText('./images/atom_fragmented.png') 

# Split fragmentation into core and fragments
fragmentation = Chem.MolToSmiles(fragmented_mol).split('.')
fragments = []
for fragment in fragmentation:
    if len([x for x in fragment if x =="*"]) ==2:
        linker = fragment
    else:
        fragments.append(fragment)
fragments = '.'.join(fragments)
linker = re.sub('[0-9]+\*', '*', linker)
fragments = re.sub('[0-9]+\*', '*', fragments)

# Get distance and angle between fragments
dist, ang = frag_utils.compute_distance_and_angle(scaff_1_sdf[0], linker, fragments)
fragments, dist, ang

# Write data to file
data_path = "./EL_data.txt"
with open(data_path, 'w') as f:
    f.write("%s %s %s %s %s" % (scaff_1_smi, linker, fragments, dist, ang))

raw_data = read_file(data_path, add_idx=True, calc_pharm_counts=True)
preprocess(raw_data, "zinc", "EL", "./", False)

# Calculate Pharmacophoric information
fragments_path = 'EL_fragments.sdf'
pharmacophores_path = 'EL_pharmacophores.sdf'
fragmentations_pharm, fails = frag_utils.create_frags_pharma_sdf_dataset([[scaff_1_smi, linker, fragments, dist, ang]], 
                                                                         scaff_1_path, dataset="CASF",
                                                                         sdffile=fragments_path,
                                                                         sdffile_pharm=pharmacophores_path,
                                                                         prot="", verbose=True)
# Write .types file
with open("EL.types", 'w') as f:
  f.write('1 ' + pharmacophores_path + ' ' + fragments_path)