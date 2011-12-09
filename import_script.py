import sys
sys.path.insert(0, '/home/floris/MPL1/lib/python2.6/site-packages/')

sys.path.append('/home/floris/src/floris_functions')
sys.path.append('/home/floris/src/analysis')
sys.path.append('/home/floris/src/flydra_analysis_tools')
sys.path.append('/home/floris/src/landing_analysis_new')

import floris_plot_lib as fpl

import numpy as np

import flydra_floris_analysis as ffa
import flydra_analysis as fa
import saccade_analysis as sac
import trajectory_plots as tap
import generate_paper_figs as gpf
import landing_paper_analysis as lpa


dataset_landing = fa.load('dataset_landing')
dataset_flyby = fa.load('dataset_flyby')
dataset_nopost = fa.load('dataset_nopost')

dataset_nopost_landing = fa.load('dataset_nopost_landing')
dataset_nopost_flyby = fa.load('dataset_nopost_flyby')
