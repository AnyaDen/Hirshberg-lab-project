#!/usr/bin/env python
# coding: utf-8

# all imports needed for the code
from matplotlib import pyplot as plt
from glob import glob
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import binom
from scipy.stats import linregress
from bokeh.plotting import figure, show, save
from bokeh.io import output_notebook, export_png
from IPython.display import display, Image
# to show bokeh graphs on jupyter
output_notebook()


# ## Functions

FOLDER_PATH = '..\\alanine dipeptide\dumps_and_logs\\'
COLORS = ['darksalmon','lavender']
FREE_ENERGY_SURFACE_IMG = '..\\alanine dipeptide\\alanine_dipeptide_free_energy_surface.png'

def create_df_angles(conformer, ensemble, steps, reduce, times, conditions,dt=0.4):
    """
    Creates datadrame of dehidral angels Psi and Phi as function of frame number.
    Reduces rows in dataframe if needed - when reduce = True
    """
    files_psi = glob('{0}{1}_{2}_{3}*dt_{4}_{5}_dihedrals_psi.dat'.format(FOLDER_PATH,conformer,ensemble,conditions,dt,steps))
    files_phi = glob('{0}{1}_{2}_{3}*dt_{4}_{5}_dihedrals_phi.dat'.format(FOLDER_PATH,conformer,ensemble,conditions,dt,steps))
    file_id = files_psi[0].split('logs\\')[1].split('_dihedral')[0]

    df_psi = pd.read_csv(files_psi[0], delimiter='\t', header=None, names=['frame','psi'])
    df_phi = pd.read_csv(files_phi[0], delimiter='\t', header=None, names=['frame','phi'])
    df_angles = pd.concat([df_psi['psi'], df_phi['phi']], axis=1)

    if reduce:
        for count in range(times):
            df_angles = df_angles.iloc[::2, :]
    
    return df_angles, file_id

def angles_vs_frame(title, dataframe, file_id):
    """
    Plots dataframe of dehidral angels Psi and Phi as function of frame number.
    """
    plot = figure(x_axis_label='frame', y_axis_label='angle', title='Dehidral Angles vs. frame - ' + file_id, plot_width=500, plot_height=500) 
    plot.dot(x = dataframe.index, y = dataframe['psi'].astype(float), legend_label = 'ψ',color='darksalmon')
    plot.dot(x = dataframe.index, y = dataframe['phi'].astype(float), legend_label = 'φ',color='rosybrown')
    show(plot)
    
def angles_2dhist(dataframe, file_id):
    """
    Plots a 2D histogram of both dehidral angles Psi and Phi.
    """
    plt.subplot()
    plt.ylabel('ψ [rad]')
    plt.xlabel('φ [rad]')
    plt.title('Dehidral Angles Histogram - ' + file_id)
    plt.hist2d(dataframe['phi'].astype(float)*np.pi/180, dataframe['psi'].astype(float)*np.pi/180, cmap = 'viridis', bins=50)
    plt.colorbar()
    plt.savefig(file_id +'_Dehidral_Angles_Histogram.png')
    plt.show()

display(Image(filename=FREE_ENERGY_SURFACE_IMG, width = 400, height = 200))


# # NVE

REDUCE = True
TIMES = 1
CONFORMER = 'c7ax'
ENSEMBLE = 'nve'
STEPS = '1B'
CONDITIONS = ''

df_angles, file_id = create_df_angles(CONFORMER, ENSEMBLE, STEPS, REDUCE, TIMES, CONDITIONS)

angles_vs_frame(CONFORMER, df_angles, file_id)

angles_2dhist(df_angles[300000:], file_id)



REDUCE = True
TIMES = 1
CONFORMER = 'c7eq'
ENSEMBLE = 'nve'
STEPS = '1B'
CONDITIONS = ''

df_angles, file_id = create_df_angles(CONFORMER, ENSEMBLE, STEPS, REDUCE, TIMES, CONDITIONS)

angles_vs_frame(CONFORMER, df_angles, file_id)

angles_2dhist(df_angles, file_id)


# # NVT - NVE + Langevin

REDUCE = False
TIMES = 1
CONFORMER = 'c7ax'
ENSEMBLE = 'nve'
STEPS = '1B'
CONDITIONS = 'langevin'

df_angles, file_id = create_df_angles(CONFORMER, ENSEMBLE, STEPS, REDUCE, TIMES, CONDITIONS)

angles_vs_frame(CONFORMER, df_angles[:30000], file_id)

angles_2dhist(df_angles[:20000], file_id)
display(Image(filename=FREE_ENERGY_SURFACE_IMG, width = 370, height = 185))


REDUCE = False
TIMES = 1
CONFORMER = 'c7eq'
ENSEMBLE = 'nve'
STEPS = '1B'
CONDITIONS = 'langevin'

df_angles, file_id = create_df_angles(CONFORMER, ENSEMBLE, STEPS, REDUCE, TIMES, CONDITIONS)

angles_vs_frame(CONFORMER, df_angles[400000:], file_id)

angles_2dhist(df_angles[450000:900000], file_id)


# # NVT - 250	Å box

REDUCE = False
TIMES = 1
CONFORMER = 'c7ax'
ENSEMBLE = 'nvt'
STEPS = '1B'
CONDITIONS = 'box250'

df_angles, file_id = create_df_angles(CONFORMER, ENSEMBLE, STEPS, REDUCE, TIMES, CONDITIONS)

angles_vs_frame(CONFORMER, df_angles, file_id)

angles_2dhist(df_angles, file_id)


# # NVT - Recenter

REDUCE = False
TIMES = 1
CONFORMER = 'c7ax'
ENSEMBLE = 'nvt'
STEPS = '1B'
CONDITIONS = 'fixrecenter'

df_angles, file_id = create_df_angles(CONFORMER, ENSEMBLE, STEPS, REDUCE, TIMES, CONDITIONS)

angles_vs_frame(CONFORMER, df_angles[:30000], file_id)
# Frame No. 7156
angles_2dhist(df_angles[:30000], file_id)



REDUCE = False
TIMES = 1
CONFORMER = 'c7eq'
ENSEMBLE = 'nvt'
STEPS = '1B'
CONDITIONS = 'fixrecenter'

df_angles, file_id = create_df_angles(CONFORMER, ENSEMBLE, STEPS, REDUCE, TIMES, CONDITIONS)

angles_vs_frame(CONFORMER, df_angles, file_id)

angles_2dhist(df_angles, file_id)


# # NVT - 0.4 dt -> 0.1 dt

REDUCE = False
TIMES = 1
CONFORMER = 'c7ax'
ENSEMBLE = 'nvt'
STEPS = '1B'
CONDITIONS = ''
DT = 0.1

df_angles, file_id = create_df_angles(CONFORMER, ENSEMBLE, STEPS, REDUCE, TIMES, CONDITIONS, DT)

angles_vs_frame(CONFORMER, df_angles, file_id)

angles_2dhist(df_angles[:400000], file_id)


REDUCE = False
TIMES = 1
CONFORMER = 'c7eq'
ENSEMBLE = 'nvt'
STEPS = '1B'
CONDITIONS = ''
DT = 0.1

df_angles, file_id = create_df_angles(CONFORMER, ENSEMBLE, STEPS, REDUCE, TIMES, CONDITIONS, DT)

angles_vs_frame(CONFORMER, df_angles, file_id)

angles_2dhist(df_angles[:400000], file_id)


# # NVT - Bigger Seed

REDUCE = False
TIMES = 1
CONFORMER = 'c7ax'
ENSEMBLE = 'nvt'
STEPS = '1B'
CONDITIONS = 'seed42587'

df_angles, file_id = create_df_angles(CONFORMER, ENSEMBLE, STEPS, REDUCE, TIMES, CONDITIONS)

angles_vs_frame(CONFORMER, df_angles, file_id)

angles_2dhist(df_angles[:400000], file_id)


# # NVT - Smaller Seed

REDUCE = False
TIMES = 1
CONFORMER = 'c7ax'
ENSEMBLE = 'nvt'
STEPS = '1B'
CONDITIONS = 'seed15213'

df_angles, file_id = create_df_angles(CONFORMER, ENSEMBLE, STEPS, REDUCE, TIMES, CONDITIONS)

angles_vs_frame(CONFORMER, df_angles, file_id)

angles_2dhist(df_angles[:400000], file_id)


# # NVT - tTamp


REDUCE = False
TIMES = 1
CONFORMER = 'c7ax'
ENSEMBLE = 'nvt'
STEPS = '1B'
CONDITIONS = 'ttamp'

df_angles, file_id = create_df_angles(CONFORMER, ENSEMBLE, STEPS, REDUCE, TIMES, CONDITIONSz

angles_vs_frame(CONFORMER, df_angles, file_id)

angles_2dhist(df_angles[:400000], file_id)



REDUCE = False
TIMES = 1
CONFORMER = 'c7eq'
ENSEMBLE = 'nvt'
STEPS = '1B'
CONDITIONS = 'ttamp'

df_angles, file_id = create_df_angles(CONFORMER, ENSEMBLE, STEPS, REDUCE, TIMES, CONDITIONS)

angles_vs_frame(CONFORMER, df_angles, file_id)

angles_2dhist(df_angles, file_id)

