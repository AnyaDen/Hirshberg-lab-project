#!/usr/bin/env python
# coding: utf-8


# all imports needed for the code
from glob import glob
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import binom
from scipy.stats import linregress
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import BoxAnnotation
# to show bokeh graphs on jupyter
output_notebook()


# ## Functions


THERMO_START = 'Step Temp TotEng KinEng PotEng \n'
THERMO_END = 'Loop time of'
COLORS = ['darksalmon', 'navy','lavender']

def separate_thermo_from_log_file(data):
    thermo = data.split(THERMO_START)[2].split(THERMO_END)[0]
    return thermo

def separate_lines(thermo):
    lines = thermo.split('\n')
    all_lines = []
    for line in lines:
        all_lines.append(line.split())
    return all_lines[:-1]

def parse_lines(all_lines):
    step = []
    temp = []
    toteng = []
    kineng = []
    poteng = []
    for line in all_lines:
        step.append(line[0])
        temp.append(line[1])
        toteng.append(line[2])
        kineng.append(line[3])
        poteng.append(line[4])
    
    thermo_df = pd.DataFrame({'step':step, 'temp':temp,'toteng':toteng,'kineng':kineng,'poteng':poteng})
    return thermo_df

def eng_change_percentage(df):
    first_energy = float(df["toteng"][0])
    df_toteng = (first_energy - df["toteng"].astype(float))*(100/first_energy)
    return df_toteng

def value_blocks_error(df, value):   
    first_value = float(df[value][0])
    df_value = df[value].astype(float)
    block_size = int(np.around(len(df_value)/4))
    tot_avg = df_value.mean()
    tot_std = df_value.std(ddof=1)
    value_1 = df_value.iloc[:block_size]
    value_2 = df_value.iloc[block_size:2*block_size]
    value_3 = df_value.iloc[2*block_size:3*block_size]
    value_4 = df_value.iloc[3*block_size:4*block_size]
    std = np.std([value_1, value_2, value_3, value_4])
    error = std/np.sqrt(4)
    return first_value, df_value, tot_avg, tot_std, error

def graph(title, dataframes, value, times = 5, eng_change = False, temp_change = True, reduce = True):
    for count, file in enumerate(dataframes.keys()):
        if eng_change:
            plot = figure(x_axis_label='step', y_axis_label='toteng change (%)', title='Change in total energy (%)', plot_width=500, plot_height=500)
            df_toteng = eng_change_percentage(list(dataframes.values())[0])
            plot.dot(x = dataframes[file]['step'].astype(float), y = df_toteng, legend_label = file.split('logs\\')[1].split('_log')[0],color=COLORS[count])
        elif temp_change:
            df = list(dataframes.values())[0]
            first_value, df_value, avg, tot_std, error = value_blocks_error(df, 'temp')
            if reduce:
                for count in range(times):
                    df_value = df_value.iloc[::2]
            plot = figure(x_axis_label='step', y_axis_label='temp [K]', title='Change in temperature', plot_width=500, plot_height=500)
            plot.dot(x = df_value.index, y = df_value, legend_label = file.split('logs\\')[1].split('_log')[0],color='rosybrown')
            plot.line(x = df_value.index, y = [first_value]*len(df_value), legend_label = 'set temp - ' + str(int(np.around(first_value))), color='firebrick')
            plot.line(x = df_value.index, y = [avg]*len(df_value), legend_label = 'avg temp - ' + str(int(np.around(avg))), color='chocolate')
            box = BoxAnnotation(bottom = avg - error, top = avg + error, fill_alpha=0.2, fill_color='red')
            plot.add_layout(box)
            print('The error is {0}\nThe total std is {1}\n'.format(error, tot_std))
        else:
            plot = figure(x_axis_label='step', y_axis_label=value, title=title, plot_width=500, plot_height=500)
            plot.dot(x = dataframes[file]['step'].astype(float), y = dataframes[file][value].astype(float), legend_label = file.split('logs\\')[1].split('_log')[0],color=COLORS[count])

    show(plot)


# # NVE


folder_path = '..\\alanine dipeptide\dumps_and_logs\\'
files = glob(folder_path + 'c7ax_nvt_seed15*_dt_0.4_1B*log.lammps')
df_dict = {}

for file in files:
    print(file)
    with open(file, 'r') as f:
        data = f.read()
    
    thermo = separate_thermo_from_log_file(data)
    all_lines = separate_lines(thermo)
    thermo_df = parse_lines(all_lines)
    df_dict[file] = thermo_df
    
graph('C7ax', df_dict,'toteng')


# # NVT

files = glob(folder_path + 'nvt*1B_log.lammps')
df_dict_nvt = {}

for file in files:
    print(file)
    with open(file, 'r') as f:
        data = f.read()
    
    thermo = separate_thermo_from_log_file(data)
    all_lines = separate_lines(thermo)
    thermo_df = parse_lines(all_lines)
    df_dict_nvt[file] = thermo_df
    
graph('C7ax',df_dict_nvt,'toteng')


# # Temperature

folder_path = '..\\alanine dipeptide\dumps_and_logs\\'
files = glob(folder_path + 'c7ax_nve_langevin*_dt_0.4_1B*log.lammps')
df_dict = {}

for file in files:
    print(file)
    with open(file, 'r') as f:
        data = f.read()
    
    thermo = separate_thermo_from_log_file(data)
    all_lines = separate_lines(thermo)
    thermo_df = parse_lines(all_lines)
    df_dict[file] = thermo_df
    
a = graph('C7ax', df_dict,'temp')

