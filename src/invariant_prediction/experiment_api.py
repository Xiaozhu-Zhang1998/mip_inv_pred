"""Compact imports shared by the manuscript experiment scripts."""
import time,random
from functools import reduce
from itertools import combinations
import numpy as np
import pandas as pd
from tqdm import tqdm
from .task1 import vanilla_msetype,precompute_stat,cd_msetype
from .task2 import vanilla_testtype,precompute_stat_test,cd_testtype
from .bootstrap import train_test_split_envs,mip_test_multiple
from .confounding import vanilla_msetype_additive
from .nonlinear import fourier_expand,spline_expand,cd_msetype_grouped
from .utils import ridge_screening,loeo_mse
from .data import *
from baselines import icp
from baselines.causal_dantzig import fit_hiddenICP
from baselines.eills import fit_eills
from baselines.negdro import negDRO
from baselines.coco import fit_coco_linear_regularized
from baselines.irm import irm_linear
from baselines.bip import pi_analytical,get_all_zs
from baselines.fair import fit_fair_nn
from baselines.wvm import DirectWassersteinVariance,WV_screening
