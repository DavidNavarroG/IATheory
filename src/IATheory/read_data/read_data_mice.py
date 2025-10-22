import numpy as np
import pandas as pd

def read_data_mice(config_setup):
    # I read the data vectors and the covariance matrix.
    path_catalogues = '/nfs/pic.es/user/d/dnavarro/IATheory/data/catalogues/'
    wgg_measured = pd.read_csv(path_catalogues + 'wgg_MICE_zs.txt', sep = ' ')
    wgp_measured = pd.read_csv(path_catalogues + 'wgp_MICE_zs.txt', sep = ' ')
    cov_mat = pd.read_csv(path_catalogues + 'cov_std_MICE_zs.txt', sep = ' ', header = None)
    cov_mat.columns = np.concatenate([wgg_measured.r.values, wgg_measured.r.values])
    cov_mat.set_index(np.concatenate([wgg_measured.r.values, wgg_measured.r.values]), inplace = True)
    
    # I apply the scale cuts
    min_rp_scale = config_setup['min_scale_cut']/0.69
    wgg_measured = wgg_measured[wgg_measured.r > min_rp_scale]
    wgp_measured = wgp_measured[wgp_measured.r > min_rp_scale]
    cov_mat = cov_mat.loc[(cov_mat.columns>min_rp_scale), (cov_mat.columns>min_rp_scale)].values
    
    # I save the transverse distance and the correlation functions
    rp_data = wgg_measured.r
    corr_wgg = wgg_measured.wgg
    corr_wgp = wgp_measured.wgp
    corr = pd.concat([corr_wgg, corr_wgp])

    return rp_data, corr, cov_mat
