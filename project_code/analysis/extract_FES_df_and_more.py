#imports
from matplotlib import pyplot as plt
from scipy import integrate
from statistics import mean, stdev
from scipy.stats import gaussian_kde as kde
import matplotlib as mpl
from glob import glob
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.constants import gas_constant
from scipy.special import binom
from scipy.stats import linregress
import natsort

PATH_TO_DIRECTORY = './pca_cos_theta/'
PATH_TO_PHI_REF = './reference/c7eq_nvt_metad_phi_dt_0.5_200ns_fes.dat'
PATH_TO_PSI_REF = './reference/c7eq_nvt_metad_psi_dt_0.5_200ns_fes.dat'
PATH_TO_PLOTS = './plots/biased/'
PATH_TO_TRANSITION_FILE = './num_of_tansitions.txt'
UNBIASED_STATS = './unbiased_CV_stats.csv'
TEMPERATURE = 300
KBT = gas_constant*TEMPERATURE/1000
DPI = 300

def moving_average(m, n=3) :
    ret = np.cumsum(m, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def calc_transitions(dataframe, metad):
    avg_dataframe = moving_average(np.array(dataframe[metad][:10000]), n=6)
    count = 0
    eq_thresh = 0.9
    ax_thresh = -2.2
#     eq_thresh = 0.95
#     ax_thresh = -2.7
    if avg_dataframe[0] > eq_thresh:
        eq = True
        ax = False
    elif avg_dataframe[0] < ax_thresh:
        eq = False
        ax = True
    else:
        eq = False
        ax = False
    for cv in avg_dataframe[1:]:
        if cv < ax_thresh:
            if eq:
                count += 1
            eq = False
            ax = True
        elif cv > eq_thresh:
            if ax:
                count += 1
            eq = True
            ax = False
        else:
            continue
    return np.round(count/2)

def find_fields_names(file):
    with open(file) as f:
        first_line = f.readline()
    
    return first_line.split('#! FIELDS ')[1].split()

def create_df(file, rows_to_skip = [0,1,2,3,4,5], columns=2):

    df_angles = pd.read_csv(file, skiprows=rows_to_skip, names=find_fields_names(file), header=None,
    						delim_whitespace=True, skip_blank_lines=True, usecols = [i for i in range(columns)])
    
    return df_angles

def get_file_id(path):
	return ' '.join(path.split('./')[1].split('/')[0].split('_'))

def fes_1d_plot(avg_df, std_df, ref_df, file_id, angle: str):
  y = avg_df.values #+ avg_df.values.min()
  plt.plot()
  plt.ylabel('Free energy [kJ/mol]')
  plt.plot(ref_df[angle], ref_df['file.free'] - ref_df['file.free'].min(), label="Reference", color='teal')
  plt.plot(avg_df.index, y, label="Reweighting Mean", color='peru')
  plt.fill_between(avg_df.index, y-std_df.values, y+std_df.values, alpha=0.5, edgecolor='chocolate', facecolor='chocolate', label="Reweighting STD")
  if angle == 'psi':
  	plt.xlabel('ψ [rad]')
  	plt.ylim(0, 25)
  	plt.legend(loc="upper right")
  elif angle == 'phi':
  	plt.xlabel('φ [rad]')
  	plt.ylim(0, 65)
  	plt.legend(loc="upper left")
  plt.title('FES for {angle} - {file_id}'.format(angle=angle, file_id=file_id))
  plt.savefig(PATH_TO_PLOTS + file_id +' FES %s comparison.png' % angle, dpi = DPI)
  plt.clf()
  

def fes_2d_plot(avg_fes, std_fes, xedges, yedges, file_id):
	xcenters = (xedges[:-1] + xedges[1:]) / 2
	ycenters = (yedges[:-1] + yedges[1:]) / 2
	plt.contourf(xcenters, ycenters, (avg_fes-avg_fes.min()).T, levels = 25, cmap = 'jet')
	plt.ylabel('ψ [rad]')
	plt.xlabel('φ [rad]')
	plt.title('2D FES - ' + file_id)
	cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=85), cmap='jet'))
	cbar.set_label('Free energy [kJ/mol]')
	plt.savefig(PATH_TO_PLOTS + file_id +'_Dihedral_Angles_FES.png', dpi = DPI)
	plt.clf()
	# plt.contourf(xcenters, ycenters, std_fes.T, levels = 25, cmap = 'binary')
	# plt.ylabel('ψ [rad]')
	# plt.xlabel('φ [rad]')
	# plt.title('2D FES STD- ' + file_id)
	# cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=4), cmap='binary'))
	# cbar.set_label('Free energy STDEV [kJ/mol]')
	# plt.savefig(PATH_TO_PLOTS + file_id +'_Dihedral_Angles_FES_STD.png', dpi = DPI)

def DeltaF_plot(times, avg_deltaf, std_deltaf, file_id):
	avg_deltaf = np.asarray(avg_deltaf)
	std_deltaf = np.asarray(std_deltaf)
	plt.plot()
	plt.ylabel(r'$\Delta$F [kJ/mol]')
	plt.axhline(y=9.9, linestyle='--' ,color='black', alpha=0.5, label='Reference')
	plt.axhline(y=8.65, linestyle='--' ,color='gray', alpha=0.25, label='± 0.5 kbt')
	plt.axhline(y=11.15, linestyle='--' ,color='gray', alpha=0.25)
	plt.plot(times, avg_deltaf, color='peru', label = r'$\Delta$F - Mean')
	plt.fill_between(times, avg_deltaf-std_deltaf, avg_deltaf+std_deltaf, alpha=0.5, color='chocolate', label=r'$\Delta$F - STD')
	plt.xlabel('Time [ns]')
	#plt.ylim(5, 15)
	plt.xticks(np.arange(0, 21, step=2))
	plt.legend(loc="upper right")
	plt.title(r'$\Delta$F - ' + file_id)
	plt.savefig(PATH_TO_PLOTS + file_id +' deltaF', dpi = DPI)
	plt.clf()

def fes_2d_reweight(list_of_files, file_id):
	fes_list = []
	for df in list_of_files:
		weights = np.exp(df['metad.bias']/KBT)
		hist, xedges, yedges = np.histogram2d(df['phi'], df['psi'], weights=weights, bins=100)
		fes = -KBT*np.log(hist)
		fes_list.append(fes)
	fes_array = np.array(fes_list)
	return fes_array.mean(axis=0), fes_array.std(axis=0), xedges, yedges

def fes_1d_average(list_of_list_of_files, angle):
	all_df = list_of_list_of_files[0][-1]
	all_df = all_df.set_index(angle)
	for list_of_files in list_of_list_of_files[1:]:
		df = list_of_files[-1]
		df = df.set_index(angle)
		all_df = pd.concat([all_df, df], axis=1)

	# all_df.replace([np.inf, -np.inf], 'NA', inplace=True)

	return all_df.mean(axis=1), all_df.std(axis=1)

def calculate_df(dataframe):
	df_values = dataframe['ffphi'] + dataframe['ffphi'].min()
	integrand = np.exp(-df_values.values/KBT)
	plus = integrate.simps(integrand[dataframe['phi'] > 0])
	minus = integrate.simps(integrand[dataframe['phi'] < 0])
	return KBT*np.log(minus/plus)

def all_df_calculations(phi_dfs):
	times = np.arange(0,19.8,0.2)
	deltafs_mean = []
	deltafs_std = []
	for i in range(99):
		time_i = []
		for phi_df in phi_dfs:
			time_i.append(calculate_df(phi_df[i]))
		deltafs_mean.append(mean(time_i))
		deltafs_std.append(stdev(time_i))
	return times, deltafs_mean, deltafs_std

def parse_reweighted_dfs(list_of_list_of_files):
	dfs = []
	for list_of_files in list_of_list_of_files:
		simulation_files = []
		for f in list_of_files:
			simulation_files.append(create_df(f))
		dfs.append(simulation_files)

	return dfs

def extract_cv_boundaries(name):
	stats = pd.read_csv(UNBIASED_STATS)
	cv_stats = stats.loc[stats['metad'] == name]
	confa_min = cv_stats['confa_min_cv'].values
	confa_max = cv_stats['confa_max_cv'].values
	confb_min = cv_stats['confb_min_cv'].values
	confb_max = cv_stats['confb_max_cv'].values
	return confa_min, confa_max, confb_min, confb_max

def cv_vs_time(dataframe, name):
	confa_min, confa_max, confb_min, confb_max = extract_cv_boundaries(name)
	plt.plot()
	plt.ylim([-1.2, 1.2])
	plt.ylabel('CV')
	plt.xlabel('Timestamp [ns]')
	plt.title(name)
	plt.plot(dataframe['time'].astype(float)/1000, dataframe['cv'], c='#ab9595', alpha=0.95)
	plt.axhspan(confa_min, confa_max, color='#F3EA6F', alpha=0.3, label='c7eq boundaries')
	plt.axhspan(confb_min, confb_max, color='teal', alpha=0.7, label='c7ax boundaries')
	plt.legend()
	plt.savefig(PATH_TO_PLOTS + name.lower() + '_cv_vs_time.png')
	plt.clf()

def phi_vs_time(dataframe, name):
	plt.plot()
	plt.plot()
	plt.ylabel(r'$\phi$')
	plt.xlabel('Timestamp [ns]')
	plt.title(name)
	plt.plot(dataframe['time'].astype(float)/1000, dataframe['phi'], c='#E9967A')
	plt.savefig(PATH_TO_PLOTS + name + '_phi_vs_time.png')
	plt.clf()

def collect_and_parse_all_files():
	phi = create_df(PATH_TO_PHI_REF, rows_to_skip=[0,1,2,3,4])
	psi = create_df(PATH_TO_PSI_REF, rows_to_skip=[0,1,2,3,4])

	phi_files = []
	psi_files = []
	for i in range(1,11):
	 	phi_files.append(natsort.natsorted(glob(PATH_TO_DIRECTORY + 'analysis*ffphi_{0}.dat'.format(i))))
	 	psi_files.append(natsort.natsorted(glob(PATH_TO_DIRECTORY + 'analysis*ffpsi_{0}.dat'.format(i))))

	phi_dfs = parse_reweighted_dfs(phi_files)
	psi_dfs = parse_reweighted_dfs(psi_files)

	dihedral_files = glob(PATH_TO_DIRECTORY + '*_dihedrals_*')
	dihedral_dfs = []
	for dihedral_file in dihedral_files:
		df = create_df(dihedral_file, rows_to_skip = [0,1,2,3,4,5,6,7], columns=10)
		dihedral_dfs.append(df)

	return phi, psi, phi_dfs, psi_dfs, dihedral_dfs

def main():
	ml_methods = ['pca_sin', 'pca_cos', 'pca_cos_theta', 'lda_sin', 'lda_cos', 'lda_cos_theta', 'hlda_sin', 'hlda_cos', 'hlda_cos_theta']
	file_id = get_file_id(PATH_TO_DIRECTORY)

	phi, psi, phi_dfs, psi_dfs, dihedral_dfs = collect_and_parse_all_files()

	### FES 1D ###

	# for phi
	avg_df, std_df = fes_1d_average(phi_dfs, 'phi')
	fes_1d_plot(avg_df, std_df, phi, file_id, 'phi')

	# for psi
	avg_df, std_df = fes_1d_average(psi_dfs, 'psi')
	fes_1d_plot(avg_df, std_df, psi, file_id, 'psi')

	### DeltaF ###
	times, deltafs_mean, deltafs_std = all_df_calculations(phi_dfs)
	DeltaF_plot(times, deltafs_mean, deltafs_std, file_id)

	### num of transitions ###
	num_of_transitions_list = []
	for dihedral_df in dihedral_dfs:
		num_of_transitions_list.append(calc_transitions(dihedral_df, 'phi'))
	print(num_of_transitions_list)
	mean_num = np.round(mean(num_of_transitions_list))

	with open(PATH_TO_TRANSITION_FILE, 'a') as f:
		f.write('{file_id},{mean_num}\n'.format(file_id=file_id, mean_num=mean_num))

	### FES 2D ###
	avg_df, std_df, xedges, yedges = fes_2d_reweight(dihedral_dfs, file_id)
	fes_2d_plot(avg_df, std_df, xedges, yedges, file_id)

	### CV vs time ###
	dihedral_df = dihedral_dfs[0]
	if PATH_TO_DIRECTORY.split('/')[1] in ml_methods:
		method = PATH_TO_DIRECTORY.split('/')[1].split('_')[0].upper()
		function = PATH_TO_DIRECTORY.split('/')[1].split('_')[1]
		name = method + '_' + function
		cv_vs_time(dihedral_df,name)
	else:
		name = PATH_TO_DIRECTORY.split('/')[1]

	### phi vs time
	phi_vs_time(dihedral_df, name)


if __name__ == '__main__':
	main()