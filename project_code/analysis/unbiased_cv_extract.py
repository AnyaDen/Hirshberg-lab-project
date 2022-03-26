from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.stats import wasserstein_distance 

SUMMARY_TEXT = 'metad,psi,phi,theta,c7ax_std,c7eq_std,distance,confa_min_cv,confa_max_cv,confb_min_cv,confb_max_cv'
UNBIASED_CONFA =  './confA/'
UNBIASED_CONFB =  './confB/'
PATH_TO_PLOTS = './plots/unbiased/'
TIME_ARRAY = np.arange(0,2,2/4001)
DPI = 300

### CV FUNCTIONS ###

# with sin
def sin_function(value):
    return np.sin(value)

# with cos
def cos_function(value):
    return np.cos(value)

# with cos and theta
def cos_theta_function(value):
    return 0.5+0.5*np.cos(value-1.25)

def find_fields_names(file):
    with open(file) as f:
        first_line = f.readline()
    
    return first_line.split('#! FIELDS ')[1].split()

def create_df(file, rows_to_skip = [0,1,2,3,4,5,6], columns=2):

    df_angles = pd.read_csv(file, skiprows=rows_to_skip, names=find_fields_names(file), header=None,
                            delim_whitespace=True, skip_blank_lines=True, usecols = [i for i in range(columns)])

    return df_angles

def scatter(confa, confb, func):
    confa_df = create_df(confa, columns = 3)
    confb_df = create_df(confb, columns = 3)
    plt.plot()
    plt.title('CV Scattering')
    plt.xlim( [ -math.pi, math.pi ] )
    plt.ylim( [ -math.pi, math.pi ] )
    if func == 'cos_theta':
        xa = confa_df[ "phi" ].apply(cos_theta_function)
        ya = confa_df[ "psi" ].apply(cos_theta_function)
        xb = confb_df[ "phi" ].apply(cos_theta_function)
        yb = confb_df[ "psi" ].apply(cos_theta_function)
        plt.scatter(xa, ya, label='c7eq',  color = 'teal', s=2)
        plt.scatter(xb, yb, label='c7ax',  color = 'peru', s=2)
        plt.legend()
        plt.xlabel( r"$0.5+0.5\dot cos(\Phi-1.25)$" )
        plt.ylabel( r"$0.5+0.5\dot cos(\Psi-1.25)$" )
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.savefig(PATH_TO_PLOTS + func + '_scattering.png', dpi=DPI)
    if func == 'cos':
        xa = confa_df[ "phi" ].apply(cos_function)
        ya = confa_df[ "psi" ].apply(cos_function)
        xb = confb_df[ "phi" ].apply(cos_function)
        yb = confb_df[ "psi" ].apply(cos_function)
        plt.scatter(xa, ya, label='c7eq',  color = 'teal', s=2)
        plt.scatter(xb, yb, label='c7ax',  color = 'peru', s=2)
        plt.legend()
        plt.xlabel( r"$cos(\Phi)$" )
        plt.ylabel( r"$cos(\Psi)$" )
        plt.xlim([-1.1, 1.1])
        plt.ylim([-1.1, 1.1])
        plt.savefig(PATH_TO_PLOTS + func + '_scattering.png', dpi=DPI)
    if func == 'sin':
        xa = confa_df[ "phi" ].apply(sin_function)
        ya = confa_df[ "psi" ].apply(sin_function)
        xb = confb_df[ "phi" ].apply(sin_function)
        yb = confb_df[ "psi" ].apply(sin_function)
        plt.scatter(xa, ya, label='c7eq',  color = 'teal', s=2)
        plt.scatter(xb, yb, label='c7ax',  color = 'peru', s=2)
        plt.legend()
        plt.xlabel( r"$sin(\Phi)$" )
        plt.ylabel( r"$sin(\Psi)$" )
        plt.xlim([-1.1, 1.1])
        plt.ylim([-1.1, 1.1])
        plt.savefig(PATH_TO_PLOTS + func + '_scattering.png', dpi=DPI)
    plt.clf()

def calc_angle_arrays(df_confa, df_confb, func):
    x1 = np.array(df_confa.drop(["time"], axis=1).apply(func)).T
    x2 = np.array(df_confb.drop(["time"], axis=1).apply(func)).T

    return x1, x2

def lda(x1, x2):
    S1 = np.cov(x1)
    S2 = np.cov(x2)

    Sw = S1 + S2

    #Means
    m1 = np.mean(x1, axis=1)
    m2 = np.mean(x2, axis=1)
    
    Sb = np.cov(np.vstack((m1, m2)).T)

    eigvals, eigvecs = np.linalg.eig(np.dot(np.linalg.inv(Sw),Sb))
    idx = eigvals.argsort() 
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]
    w = eigvecs[:,-1]

    return [abs(np.real(x)) for x in w]

def hist(sconfa, sconfb, func, method = 'HLDA'):
    plt.plot()
    plt.ylabel('Count')
    plt.xlabel('CV')
    plt.title(method + ' CV Histogram')
    plt.hist(sconfa, bins=100, ls='dashed', alpha = 0.7, lw=2, color='teal', label='c7eq',density=True)
    plt.hist(sconfb, bins = 100, ls='dashed', alpha = 0.7, lw=2, color='peru', label='c7ax',density=True)
    plt.legend()
    plt.savefig(PATH_TO_PLOTS + method + '_' + func + '_CV_Histogram.png', dpi=DPI)
    plt.clf()

def hlda(x1, x2):
    S1 = np.cov(x1)
    S2 = np.cov(x2)

    #Means
    m1 = np.mean(x1, axis=1)
    m2 = np.mean(x2, axis=1)

    Sb = np.cov(np.vstack((m1, m2)).T)

    Sw_inv = np.linalg.inv(S1) + np.linalg.inv(S2) 

    eigvals, eigvecs = np.linalg.eig(np.dot(Sw_inv,Sb))

    w = eigvecs[:, eigvals>1E-6]

    w_list = []
    for i in w:
        w_list.append(abs(np.real(i[0])))

    return w_list

def width_and_distance(sconfa, sconfb):
    sconfa_hist, confa_bins = np.histogram(sconfa, bins=100)
    sconfb_hist, confb_bins = np.histogram(sconfb, bins=100)

    confa_mids = 0.5*(confa_bins[1:] + confa_bins[:-1])
    confb_mids = 0.5*(confb_bins[1:] + confb_bins[:-1])

    confa_mean = np.average(confa_mids, weights=sconfa_hist)
    confb_mean = np.average(confb_mids, weights=sconfb_hist)

    confa_var = np.average((confa_mids - confa_mean)**2, weights=sconfa_hist)
    confb_var = np.average((confb_mids - confb_mean)**2, weights=sconfb_hist)

    confa_std = np.sqrt(confa_var)
    confb_std = np.sqrt(confb_var)
    distance = abs(confa_mean - confb_mean)/np.sqrt(confa_std+confb_std)

    return confa_std, confb_std, distance

def pca(x1, x2):
    angles_array = np.concatenate((x1, x2), axis = 1)
    c = np.cov(angles_array)
    e, v = np.linalg.eig(c)
    # coefficients
    w = v[:,0]
    return list(w)

def method_calc(method, method_name, confa, confb, func, func_name, num):
    values_dict = {}
    df_confa = create_df(confa, columns=4)
    df_confb = create_df(confb, columns=4)
    x1, x2 = calc_angle_arrays(df_confa, df_confb, func)
    w = method(x1, x2)

    values_dict['coeff_list'] = w

    smethod_c7eq = np.dot(np.array(w).T, x1 )
    smethod_c7ax = np.dot(np.array(w).T, x2 )

    smethod_c7eq = smethod_c7eq.reshape(-1)
    smethod_c7ax = smethod_c7ax.reshape(-1)

    values_dict['min_a'] = smethod_c7eq.min()
    values_dict['max_a'] = smethod_c7eq.max()
    values_dict['min_b'] = smethod_c7ax.min()
    values_dict['max_b'] = smethod_c7ax.max()

    confa_std, confb_std, distance = width_and_distance(smethod_c7eq, smethod_c7ax)
    values_dict['confa_std'] = confa_std
    values_dict['confb_std'] = confb_std
    values_dict['distance'] = distance

    if num==0:
        plt.plot()
        plt.xlabel('CV')
        plt.ylabel('Time [ns]')
        plt.plot(TIME_ARRAY, smethod_c7eq, label='c7eq', color = 'teal')
        plt.plot(TIME_ARRAY, smethod_c7ax, label='c7ax', color = 'peru')
        plt.legend()
        plt.savefig(PATH_TO_PLOTS + method_name + '_%s_cv.png' % func_name, dpi=DPI)
        plt.clf()

        if method_name == 'HLDA' and func_name == 'cos_theta':
            hist(smethod_c7eq, smethod_c7ax, func_name)

    return values_dict


def method_mean_calc(method, method_name, confa_files, confb_files, func, func_name):
    sum_dict = {'coeff_list':[],'confa_std':[],'confb_std':[], 'distance':[], 'min_a':[], 'max_a':[], 'min_b':[], 'max_b':[]}
    for i in range(len(confa_files)):
        values_dict =  method_calc(method, method_name, confa_files[i], confb_files[i], func, func_name, i)
        for value in values_dict:
            sum_dict[value].append(values_dict[value])

    for attrib in sum_dict:
        sum_dict[attrib] = np.mean(sum_dict[attrib], axis=0)
        if attrib == 'coeff_list':
            coeff_mean = ','.join([str(i) for i in sum_dict[attrib]])

    return sum_dict

def main():
    function_dict = {'sin':sin_function, 'cos':cos_function, 'cos_theta':cos_theta_function}
    method_dict = {'PCA':pca, 'LDA':lda, 'HLDA':hlda}

    confa_files = glob(UNBIASED_CONFA + '*_dihedrals_*')
    confb_files = glob(UNBIASED_CONFB + '*_dihedrals_*')

    ## General - Scattering ##
    for func in function_dict: 
        scatter(confa_files[0], confb_files[0], func)

    file_str = SUMMARY_TEXT

    for method in method_dict:
        print('working on ' + method)
        for func in function_dict:
            print('working on ' + func)
            values_dict = method_mean_calc(method_dict[method], method, confa_files, confb_files, function_dict[func], func)
            file_str += '\n{0}_{1},{2},{3},{4},{5},{6},{7},{8},{9}'.format(method,func,','.join(str(x) for x in values_dict['coeff_list']),values_dict['confa_std'],values_dict['confb_std'],values_dict['distance'],values_dict['min_a'],values_dict['max_a'],values_dict['min_b'],values_dict['max_b'])


    with open('unbiased_CV_stats.csv', 'w') as f:
        f.write(file_str)

if __name__ == '__main__':
    main()