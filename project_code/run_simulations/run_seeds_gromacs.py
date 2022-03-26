### IMPORTS ###
import os
import numpy as np
import pandas as pd
from glob import glob

STEPS = 40000000

### CODE ###

# importing the template for the plumed dat file
with open("dihedral_angles_analysis_input_template.dat", 'r') as f:
	plumed_template = f.read()

i = 0

topol_tpr_files = glob('topol_*.tpr')

for topol_tpr in topol_tpr_files:
	i += 1

	dihedral_angles_analysis_input = plumed_template.format(i = i)

	with open('dihedral_angles_analysis_input_{i}.dat'.format(i = str(i)), 'w') as f:
		f.write(dihedral_angles_analysis_input)

	os.system('gmx mdrun -s {tpr_file} -plumed dihedral_angles_analysis_input_{i}.dat -nsteps {steps}'.format(tpr_file = topol_tpr,i = str(i), steps = STEPS))