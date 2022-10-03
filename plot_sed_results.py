import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from scipy.constants import c
from astropy import units as u
from astropy.coordinates import SkyCoord

import matplotlib.image as mpimg

from visualization_helper import VisualizeHelper as vh

# the galaxy of interest is NGC 7496
ra_cutout = 347.43617212815616
dec_cutout = -43.41496001628405
cutout_size = (2, 2)  # in arc seconds
circle_rad = 0.2  # in arcseconds
cutout_pos = SkyCoord(ra=ra_cutout, dec=dec_cutout, unit=(u.degree, u.degree), frame='fk5')

# all the filters used for cigale fit
filter_list = np.array(['F275W', 'F336W', 'F438W', 'F555W', 'F814W',
                        'F200W', 'F300M', 'F335M', 'F360M',
                        'F770W', 'F1000W', 'F1130W', 'F2100W'])

# the order of filters in the cigale flux file
cigale_flux_file_col_name_ord = np.array(['id', 'redshift',
                                          'F275W', 'F275W_err', 'F336W', 'F336W_err', 'F438W', 'F438W_err',
                                          'F555W', 'F555W_err', 'F814W', 'F814W_err', 'F200W', 'F200W_err',
                                          'F300M', 'F300M_err', 'F335M', 'F335M_err',  'F360M', 'F360M_err',
                                          'F770W', 'F770W_err', 'F1000W', 'F1000W_err', 'F1130W', 'F1130W_err',
                                          'F2100W', 'F2100W_err'])
cigale_filt_names = np.array(['F275W_UVIS_CHIP2', 'F336W_UVIS_CHIP2', 'F438W_UVIS_CHIP2', 'F555W_UVIS_CHIP2',
                              'F814W_UVIS_CHIP2', 'jwst.nircam.F200W', 'jwst.nircam.F300M', 'jwst.nircam.F335M',
                              'jwst.nircam.F360M', 'jwst.miri.F770W', 'jwst.miri.F1000W', 'jwst.miri.F1130W',
                              'jwst.miri.F2100W'])
hdu_best_model_file_name = 'out/emb_clu_best_model.fits'
flux_fitted_model_file_name = 'out/results.fits'
# only if there is a cigale version on your computer otherwise leave it as None
cigale_logo_file_name = '/home/benutzer/software/cigale-ssp/pcigale_plots/resources/CIGALE.png'

# get the cigale flux file for the observed data
cigale_flux_file_name = 'data/flux_file_example.dat'
# cigale_flux_file_name = 'flux_file_many_lines.dat'
# specify the names in the file header to get the position of the filters
# index in the cigale flux_file
index_cigale_table = 0

# destination to the HST and JWST files
folder_hst = '/home/benutzer/data/PHANGS-HST/ngc7496/ngc7496_mosaics_01sep2020/'
folder_jwst = '/home/benutzer/data/PHANGS-JWST/ngc7496/'

file_name_hst_f275w = folder_hst + 'hlsp_phangs-hst_hst_wfc3-uvis_ngc7496_f275w_v1_exp-drc-sci.fits'
file_name_hst_f336w = folder_hst + 'hlsp_phangs-hst_hst_wfc3-uvis_ngc7496_f336w_v1_exp-drc-sci.fits'
file_name_hst_f438w = folder_hst + 'hlsp_phangs-hst_hst_wfc3-uvis_ngc7496_f438w_v1_exp-drc-sci.fits'
file_name_hst_f555w = folder_hst + 'hlsp_phangs-hst_hst_wfc3-uvis_ngc7496_f555w_v1_exp-drc-sci.fits'
file_name_hst_f814w = folder_hst + 'hlsp_phangs-hst_hst_wfc3-uvis_ngc7496_f814w_v1_exp-drc-sci.fits'

file_name_jwst_f200w = folder_jwst + 'ngc7496_nircam_lv3_f200w_i2d_align.fits'
file_name_jwst_f300m = folder_jwst + 'ngc7496_nircam_lv3_f300m_i2d_align.fits'
file_name_jwst_f335m = folder_jwst + 'ngc7496_nircam_lv3_f335m_i2d_align.fits'
file_name_jwst_f360m = folder_jwst + 'ngc7496_nircam_lv3_f360m_i2d_align.fits'

file_name_jwst_f770w = folder_jwst + 'ngc7496_miri_f770w_anchored_at2100.fits'
file_name_jwst_f1000w = folder_jwst + 'ngc7496_miri_f1000w_anchored_at2100.fits'
file_name_jwst_f1130w = folder_jwst + 'ngc7496_miri_f1130w_anchored_at2100.fits'
file_name_jwst_f2100w = folder_jwst + 'ngc7496_miri_f2100w_anchored_at2100.fits'


hst_file_name_list = [file_name_hst_f275w, file_name_hst_f336w, file_name_hst_f438w, file_name_hst_f555w,
                      file_name_hst_f814w]
nircam_file_name_list = [file_name_jwst_f200w, file_name_jwst_f300m, file_name_jwst_f335m, file_name_jwst_f360m]
miri_file_name_list = [file_name_jwst_f770w, file_name_jwst_f1000w, file_name_jwst_f1130w, file_name_jwst_f2100w]


fig = vh.plot_cigale_sed_results(cigale_flux_file_name=cigale_flux_file_name,
                                 hdu_best_model_file_name=hdu_best_model_file_name,
                                 flux_fitted_model_file_name=flux_fitted_model_file_name,
                                 hst_file_name_list=hst_file_name_list, nircam_file_name_list=nircam_file_name_list,
                                 miri_file_name_list=miri_file_name_list, cutout_pos=cutout_pos,
                                 cutout_size=cutout_size, hdu_number_hst=0, hdu_number_nircam='SCI', hdu_number_miri=0,
                                 filter_list=filter_list, cigale_flux_file_col_name_ord=cigale_flux_file_col_name_ord,
                                 index_cigale_table=index_cigale_table, cigale_filt_names=cigale_filt_names,
                                 cigale_logo_file_name=cigale_logo_file_name, circle_rad=circle_rad)

# plt.show()
fig.savefig('plot_output/sed_total.png')
fig.savefig('plot_output/sed_total.pdf')

