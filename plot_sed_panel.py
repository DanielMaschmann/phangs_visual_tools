import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

from astropy.io import fits
from visualization_helper import VisualizeHelper as vh




folder_hst = '/home/benutzer/data/PHANGS-HST/ngc1365/'
folder_jwst = '/home/benutzer/data/PHANGS-JWST/ngc1365/'
folder_alma = '/home/benutzer/data/PHANGS-ALMA/ngc1365/'
folder_muse = '/home/benutzer/data/PHANGS-MUSE/ngc1365/'

file_name_hst_f275w = folder_hst + 'hlsp_phangs-hst_hst_wfc3-uvis_ngc1365_f275w_v1_exp-drc-sci.fits'
file_name_hst_f336w = folder_hst + 'hlsp_phangs-hst_hst_wfc3-uvis_ngc1365_f336w_v1_exp-drc-sci.fits'
file_name_hst_f438w = folder_hst + 'hlsp_phangs-hst_hst_wfc3-uvis_ngc1365_f438w_v1_exp-drc-sci.fits'
file_name_hst_f555w = folder_hst + 'hlsp_phangs-hst_hst_wfc3-uvis_ngc1365_f555w_v1_exp-drc-sci.fits'
file_name_hst_f814w = folder_hst + 'hlsp_phangs-hst_hst_wfc3-uvis_ngc1365_f814w_v1_exp-drc-sci.fits'

file_name_hst_f275w_err = folder_hst + 'hlsp_phangs-hst_hst_wfc3-uvis_ngc1365_f275w_v1_err-drc-wht.fits'
file_name_hst_f336w_err = folder_hst + 'hlsp_phangs-hst_hst_wfc3-uvis_ngc1365_f336w_v1_err-drc-wht.fits'
file_name_hst_f438w_err = folder_hst + 'hlsp_phangs-hst_hst_wfc3-uvis_ngc1365_f438w_v1_err-drc-wht.fits'
file_name_hst_f555w_err = folder_hst + 'hlsp_phangs-hst_hst_wfc3-uvis_ngc1365_f555w_v1_err-drc-wht.fits'
file_name_hst_f814w_err = folder_hst + 'hlsp_phangs-hst_hst_wfc3-uvis_ngc1365_f814w_v1_err-drc-wht.fits'


file_name_jwst_f200w = folder_jwst + 'ngc1365_nircam_lv3_f200w_i2d_align.fits'
file_name_jwst_f300m = folder_jwst + 'ngc1365_nircam_lv3_f300m_i2d_align.fits'
file_name_jwst_f335m = folder_jwst + 'ngc1365_nircam_lv3_f335m_i2d_align.fits'
file_name_jwst_f360m = folder_jwst + 'ngc1365_nircam_lv3_f360m_i2d_align.fits'

file_name_jwst_f770w = folder_jwst + 'ngc1365_miri_f770w_anchored_fixedsatur_cutout.sci.dzliu.fits'
file_name_jwst_f1000w = folder_jwst + 'ngc1365_miri_f1000w_anchored_fixedsatur_cutout.sci.dzliu.fits'
file_name_jwst_f1130w = folder_jwst + 'ngc1365_miri_f1130w_anchored_fixedsatur_cutout.sci.dzliu.fits'
file_name_jwst_f2100w = folder_jwst + 'ngc1365_miri_f2100w_anchored_fixedsatur_cutout.sci.dzliu.fits'
#
# file_name_jwst_f770w = folder_jwst + 'ngc1365_miri_f770w_anchored.fits'
# file_name_jwst_f1000w = folder_jwst + 'ngc1365_miri_f1000w_anchored.fits'
# file_name_jwst_f1130w = folder_jwst + 'ngc1365_miri_f1130w_anchored.fits'
# file_name_jwst_f2100w = folder_jwst + 'ngc1365_miri_f2100w_anchored.fits'

file_name_jwst_f770w_err = folder_jwst + 'ngc1365_miri_f770w_noisemap.fits'
file_name_jwst_f1000w_err = folder_jwst + 'ngc1365_miri_f1000w_noisemap.fits'
file_name_jwst_f1130w_err = folder_jwst + 'ngc1365_miri_f1130w_noisemap.fits'
file_name_jwst_f2100w_err = folder_jwst + 'ngc1365_miri_f2100w_noisemap.fits'

channel_hst_f275w = 'f275w'
channel_hst_f336w = 'f336w'
channel_hst_f438w = 'f438w'
channel_hst_f555w = 'f555w'
channel_hst_f814w = 'f814w'

channel_jwst_f200w = 'f200w'
channel_jwst_f300m = 'f300m'
channel_jwst_f335m = 'f335m'
channel_jwst_f360m = 'f360m'

channel_jwst_f770w = 'f770w'
channel_jwst_f1000w = 'f1000w'
channel_jwst_f1130w = 'f1130w'
channel_jwst_f2100w = 'f2100w'


conv_fact_f275w = fits.open(file_name_hst_f275w)[0].header['PHOTFNU']
conv_fact_f336w = fits.open(file_name_hst_f336w)[0].header['PHOTFNU']
conv_fact_f438w = fits.open(file_name_hst_f438w)[0].header['PHOTFNU']
conv_fact_f555w = fits.open(file_name_hst_f555w)[0].header['PHOTFNU']
conv_fact_f814w = fits.open(file_name_hst_f814w)[0].header['PHOTFNU']

hst_file_name_list = [file_name_hst_f275w, file_name_hst_f336w, file_name_hst_f438w, file_name_hst_f555w, file_name_hst_f814w]
hst_err_files = [file_name_hst_f275w_err, file_name_hst_f336w_err, file_name_hst_f438w_err, file_name_hst_f555w_err,
                 file_name_hst_f814w_err]
hst_channel_list = [channel_hst_f275w, channel_hst_f336w, channel_hst_f438w, channel_hst_f555w, channel_hst_f814w]
nircam_file_name_list = [file_name_jwst_f200w, file_name_jwst_f300m, file_name_jwst_f335m, file_name_jwst_f360m]
nircam_channel_list = [channel_jwst_f200w, channel_jwst_f300m, channel_jwst_f335m, channel_jwst_f360m]
miri_file_name_list = [file_name_jwst_f770w, file_name_jwst_f1000w, file_name_jwst_f1130w, file_name_jwst_f2100w]
miri_err_file_name_list = [file_name_jwst_f770w_err, file_name_jwst_f1000w_err, file_name_jwst_f1130w_err, file_name_jwst_f2100w_err]
miri_channel_list = [channel_jwst_f770w, channel_jwst_f1000w, channel_jwst_f1130w, channel_jwst_f2100w]

cluster_list = np.genfromtxt('data/ngc1365_for_daniel_sep_30_2022.txt', dtype=object)
names = cluster_list[0]
cluster_list[cluster_list == b'""'] = b'nan'
data = np.array(cluster_list[1:], dtype=float)

ra = data[:, names == b'raj2000']
dec = data[:, names == b'dej2000']

flux_F275W = data[:, names == b'flux_F275W_5'] * conv_fact_f275w * 1e3
flux_F336W = data[:, names == b'flux_F336W_5'] * conv_fact_f336w * 1e3
flux_F438W = data[:, names == b'flux_F438W_5'] * conv_fact_f438w * 1e3
flux_F555W = data[:, names == b'flux_F555W_5'] * conv_fact_f555w * 1e3
flux_F814W = data[:, names == b'flux_F814W_5'] * conv_fact_f814w * 1e3
flux_F200W = data[:, names == b'flux_F200W_5'] * 1e9
flux_F300M = data[:, names == b'flux_F300M_5'] * 1e9
flux_F335M = data[:, names == b'flux_F335M_5'] * 1e9
flux_F360M = data[:, names == b'flux_F360M_5'] * 1e9
flux_F770W = data[:, names == b'flux_F770W_5'] * 1e9
flux_F1000W = data[:, names == b'flux_F1000W_5'] * 1e9
flux_F1130W = data[:, names == b'flux_F1130W_5'] * 1e9
flux_F2100W = data[:, names == b'flux_F2100W_5'] * 1e9

flux_F275W_err = data[:, names == b'er_flux_F275W_5'] * conv_fact_f275w * 1e3
flux_F336W_err = data[:, names == b'er_flux_F336W_5'] * conv_fact_f336w * 1e3
flux_F438W_err = data[:, names == b'er_flux_F438W_5'] * conv_fact_f438w * 1e3
flux_F555W_err = data[:, names == b'er_flux_F555W_5'] * conv_fact_f555w * 1e3
flux_F814W_err = data[:, names == b'er_flux_F814W_5'] * conv_fact_f814w * 1e3
flux_F200W_err = data[:, names == b'er_flux_F200W_5'] * 1e9
flux_F300M_err = data[:, names == b'er_flux_F300M_5'] * 1e9
flux_F335M_err = data[:, names == b'er_flux_F335M_5'] * 1e9
flux_F360M_err = data[:, names == b'er_flux_F360M_5'] * 1e9
flux_F770W_err = data[:, names == b'er_flux_F770W_5'] * 1e9
flux_F1000W_err = data[:, names == b'er_flux_F1000W_5'] * 1e9
flux_F1130W_err = data[:, names == b'er_flux_F1130W_5'] * 1e9
flux_F2100W_err = data[:, names == b'er_flux_F2100W_5'] * 1e9

flux_list = np.array([flux_F275W, flux_F336W, flux_F438W, flux_F555W, flux_F814W, flux_F200W, flux_F300M, flux_F335M, flux_F360M, flux_F770W, flux_F1000W, flux_F1130W, flux_F2100W])
flux_err_list = np.array([flux_F275W_err, flux_F336W_err, flux_F438W_err, flux_F555W_err, flux_F814W_err, flux_F200W_err, flux_F300M_err, flux_F335M_err, flux_F360M_err, flux_F770W_err, flux_F1000W_err, flux_F1130W_err, flux_F2100W_err])

# possibles apertures in arc seconds
ap_list = [0.031, 0.062, 0.093, 0.124, 0.155, 0.186]
aperture_rad = ap_list[4]


list_index = 0
cluster_index = 2
cutout_pos = SkyCoord(ra=ra[cluster_index], dec=dec[cluster_index], unit=(u.degree, u.degree), frame='fk5')
best_model_file = 'out/%i_best_model.fits' % cluster_index

fig = vh.plot_cigale_sed_panel(cigale_flux_file_name='flux_file.dat',
                               hdu_best_model_file_name=best_model_file,
                               flux_fitted_model_file_name='out/results.fits',
                               hst_file_name_list=hst_file_name_list, nircam_file_name_list=nircam_file_name_list,
                               miri_file_name_list=miri_file_name_list,
                               cutout_pos=cutout_pos, cutout_size=(2, 2),
                               hdu_number_hst=0, hdu_number_nircam='SCI', hdu_number_miri=0,
                               filter_list=None, cigale_flux_file_col_name_ord=None, index_cigale_table=list_index,
                               cigale_filt_names=None, cigale_logo_file_name=None, circle_rad=aperture_rad,
                               filter_colors=None)

fig.savefig('plot_output/sed_panel_%i' % cluster_index)





