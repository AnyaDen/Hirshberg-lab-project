### IMPORTS ###
import os
import numpy as np
import pandas as pd
from scipy.constants import gas_constant

def find_fields_names(file):
    with open(file) as f:
        first_line = f.readline()
    
    return first_line.split('#! FIELDS ')[1].split()

def create_df_angles_plumed_metad(f):
    """
    Creates datadrame of dihedral angels Psi and Phi as function of frame number for a plumed output file.
    Reduces rows in dataframe if needed - when reduce = True
    """
    df_angles = pd.read_csv(f, skiprows=[0,1,2,3,4,5,6,7], names=find_fields_names(f), delim_whitespace=True, skip_blank_lines=True)

    #df_angles = df_angles.iloc[: , 1:]
    return df_angles

### CODE ###

# 10 different seeds
seed_list = ['28459','30000','967','47691', '355', '5863', '1934', '58761', '8594', '74092']

# importing the template for the lammps input file
with open("alanine_dipeptide_c7eq_LAMMPS_template.input", 'r') as f:
	lammps_input_template = f.read()

# importing the template for the plumed dat file
with open("dihedral_angles_analysis_input_template.dat", 'r') as f:
	plumed_template = f.read()

i = 0

for seed in seed_list:
	i += 1
	
	lammps_input = lammps_input_template.format(i = i, seed = seed)

	with open('alanine_dipeptide_c7eq_LAMMPS_{i}.input'.format(i = str(i)), 'w') as f:
		f.write(lammps_input)

	dihedral_angles_analysis_input = plumed_template.format(i = i)

	with open('dihedral_angles_analysis_input_{i}.dat'.format(i = str(i)), 'w') as f:
		f.write(dihedral_angles_analysis_input)

	os.system('../../lammps/src/lmp_serial -in alanine_dipeptide_c7eq_LAMMPS_{i}.input'.format(i = str(i)))