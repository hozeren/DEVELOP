#!/usr/bin/env python

import sys
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
import data.prepare_data_linker_design
from data.prepare_data_linker_design import read_file, preprocess

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

# Arguments for DeLinker
args = defaultdict(None)
args['--dataset'] = 'zinc'
args['--config'] = '{"generation": true, \
                     "batch_size": 1, \
                     "number_of_generation_per_valid": 1000, \
                     "min_atoms": 1, "max_atoms": 5, \
                     "train_file": "molecules_EL.json", \
                     "valid_file": "molecules_EL.json", \
                     "train_struct_file": "./EL.types", \
                     "valid_struct_file": "./EL.types", \
                     "struct_data_root": "./", \
                     "output_name": "EL.smi"}'
args['--freeze-graph-model'] = False
args['--restore'] = '/home/kevin/Desktop/DEVELOP/models/scaffold_elaboration/pretrained_DEVELOP_model.pickle'

# Setup model and generate molecules
model = DenseGGNNChemModel(args)
model.train()
# Free up some memory
model = ''

"""
                     "min_atoms": 5, "max_atoms": 10, \
                        """