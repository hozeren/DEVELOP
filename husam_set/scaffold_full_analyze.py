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

from rdkit.Chem.Draw import IPythonConsole

import os
from collections import defaultdict

IPythonConsole.ipython_useSVG = True

import DEVELOP
from DEVELOP import DenseGGNNChemModel
import scaffold_hopping.scaffold_2D
from scaffold_hopping.scaffold_3D_class import run3D
from scaffold_hopping.scaffold_analysis import run_analysis

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

if use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]= "0"

#argument for reference molecules
global scaff_1_path
scaff_1_path = './celecoxib.sdf'

# Arguments for DeLinker
args = defaultdict(None)
args['--dataset'] = 'zinc'
args['--config'] = '{"generation": true, \
                     "batch_size": 1, \
                     "number_of_generation_per_valid": 250, \
                     "train_file": "molecules_scaffold_hopping.json", \
                     "valid_file": "molecules_scaffold_hopping.json", \
                     "train_struct_file": "./scaffold_hopping.types", \
                     "valid_struct_file": "./scaffold_hopping.types", \
                     "struct_data_root": "./", \
                     "output_name": "scaffold_hopping.smi"}'
args['--freeze-graph-model'] = False
args['--restore'] = '/home/kevin/Desktop/DEVELOP/models/linker_design/pretrained_DEVELOP_model.pickle'

# Setup model and generate molecules
model = DenseGGNNChemModel(args)
model.train()
# Free up some memory
model = ''

names_frags, best_scores_frags, best_rmsd_frags, names_rmsd_frags = run3D(scaff_1_path, n_cores, use_gpu)
scaffold_hopping.scaffold_analysis.run_analysis(names_frags, best_scores_frags, best_rmsd_frags, names_rmsd_frags)


"""
                     "min_atoms": 5, "max_atoms": 10, \
                        """

















