import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

from astropy.io import fits
from visualization_helper import VisualizeHelper as vh



folder_hst = '/home/benutzer/data/PHANGS-HST/ngc7496/ngc7496_mosaics_01sep2020/'
folder_jwst = '/home/benutzer/data/PHANGS-JWST/ngc7496/'


file_name_hst_f275w = folder_hst + 'hlsp_phangs-hst_hst_wfc3-uvis_ngc7496_f275w_v1_exp-drc-sci.fits'
file_name_hst_f336w = folder_hst + 'hlsp_phangs-hst_hst_wfc3-uvis_ngc7496_f336w_v1_exp-drc-sci.fits'
file_name_hst_f438w = folder_hst + 'hlsp_phangs-hst_hst_wfc3-uvis_ngc7496_f438w_v1_exp-drc-sci.fits'
file_name_hst_f555w = folder_hst + 'hlsp_phangs-hst_hst_wfc3-uvis_ngc7496_f555w_v1_exp-drc-sci.fits'
file_name_hst_f814w = folder_hst + 'hlsp_phangs-hst_hst_wfc3-uvis_ngc7496_f814w_v1_exp-drc-sci.fits'


file_name_hst_f275w_err = folder_hst + 'hlsp_phangs-hst_hst_wfc3-uvis_ngc7496_f275w_v1_err-drc-wht.fits'
file_name_hst_f336w_err = folder_hst + 'hlsp_phangs-hst_hst_wfc3-uvis_ngc7496_f336w_v1_err-drc-wht.fits'
file_name_hst_f438w_err = folder_hst + 'hlsp_phangs-hst_hst_wfc3-uvis_ngc7496_f438w_v1_err-drc-wht.fits'
file_name_hst_f555w_err = folder_hst + 'hlsp_phangs-hst_hst_wfc3-uvis_ngc7496_f555w_v1_err-drc-wht.fits'
file_name_hst_f814w_err = folder_hst + 'hlsp_phangs-hst_hst_wfc3-uvis_ngc7496_f814w_v1_err-drc-wht.fits'


file_name_jwst_f200w = folder_jwst + 'ngc7496_nircam_lv3_f200w_i2d_align.fits'
file_name_jwst_f300m = folder_jwst + 'ngc7496_nircam_lv3_f300m_i2d_align.fits'
file_name_jwst_f335m = folder_jwst + 'ngc7496_nircam_lv3_f335m_i2d_align.fits'
file_name_jwst_f360m = folder_jwst + 'ngc7496_nircam_lv3_f360m_i2d_align.fits'

file_name_jwst_f770w = folder_jwst + 'ngc7496_miri_f770w_anchored.fits'
file_name_jwst_f1000w = folder_jwst + 'ngc7496_miri_f1000w_anchored.fits'
file_name_jwst_f1130w = folder_jwst + 'ngc7496_miri_f1130w_anchored.fits'
file_name_jwst_f2100w = folder_jwst + 'ngc7496_miri_f2100w_anchored.fits'

file_name_jwst_f770w_err = folder_jwst + 'ngc7496_miri_f770w_noisemap.fits'
file_name_jwst_f1000w_err = folder_jwst + 'ngc7496_miri_f1000w_noisemap.fits'
file_name_jwst_f1130w_err = folder_jwst + 'ngc7496_miri_f1130w_noisemap.fits'
file_name_jwst_f2100w_err = folder_jwst + 'ngc7496_miri_f2100w_noisemap.fits'

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

cluster_list = np.genfromtxt('../data/candidates_embd_clus_v2p1.txt', dtype=object)
names = cluster_list[0]
cluster_list[cluster_list == b'""'] = b'nan'
data = np.array(cluster_list[1:], dtype=float)

ra = data[:, names == b'raj2000']
dec = data[:, names == b'dej2000']


hst_ap_list = np.array(vh.get_50p_aperture(instrument='hst'))
nircam_ap_list = np.array(vh.get_50p_aperture(instrument='nircam')) * 1.5
miri_ap_list = np.array(vh.get_50p_aperture(instrument='miri')) * 1.3

ap_list = np.concatenate([hst_ap_list, nircam_ap_list, miri_ap_list])


list_index = 0
cluster_index_list = [53]
for cluster_index in cluster_index_list:
    cutout_pos = SkyCoord(ra=ra[cluster_index], dec=dec[cluster_index], unit=(u.degree, u.degree), frame='fk5')
    best_model_file = 'out/%i_best_model.fits' % cluster_index

    flux_dict = vh.get_multiband_flux_from_circ_ap(cutout_pos=cutout_pos,
                                                   hst_list=hst_file_name_list, hst_err_list=hst_err_files,
                                                   hst_ap_rad_list=hst_ap_list,
                                                   nircam_list=nircam_file_name_list, nircam_ap_rad_list=nircam_ap_list,
                                                   miri_list=miri_file_name_list, miri_err_list=miri_err_file_name_list,
                                                   miri_ap_rad_list=miri_ap_list,
                                                   hst_channel_list=hst_channel_list, nircam_channel_list=nircam_channel_list,
                                                   miri_channel_list=miri_channel_list,
                                                   cutout_fact_hst=4, cutout_fact_nircam=4, cutout_fact_miri=4,
                                                   cutout_re_center_hst=4, cutout_re_center_nircam=2, cutout_re_center_miri=2,
                                                   re_center_peak_hst=False, re_center_peak_nircam=True, re_center_peak_miri=True,
                                                   hdu_number_hst=0, hdu_number_nircam='SCI', hdu_number_miri=0,
                                                   hdu_err_number_hst=0, hdu_err_number_nircam='ERR', hdu_err_number_miri=0,
                                                   flux_unit='mJy')
    position_list = []
    for channel in hst_channel_list:
        position_list.append(flux_dict['position_%s' % channel])
    for channel in nircam_channel_list:
        position_list.append(flux_dict['position_%s' % channel])
    for channel in miri_channel_list:
        position_list.append(flux_dict['position_%s' % channel])

    full_sed_fig = vh.plot_cigale_sed_results(cigale_flux_file_name='flux_file.dat',
                                     hdu_best_model_file_name=best_model_file,
                                     flux_fitted_model_file_name='out/results.fits',
                                     hst_file_name_list=hst_file_name_list, nircam_file_name_list=nircam_file_name_list,
                                     miri_file_name_list=miri_file_name_list,
                                     cutout_size=(2, 2), cutout_pos=cutout_pos, cutout_pos_list=position_list,
                                     hdu_number_hst=0, hdu_number_nircam='SCI', hdu_number_miri=0,
                                     filter_list=None, cigale_flux_file_col_name_ord=None, index_cigale_table=list_index,
                                     cigale_filt_names=None, cigale_logo_file_name=None, circle_rad=ap_list,
                                     filter_colors=None)

    full_sed_fig.savefig('plot_output/sed_%i' % cluster_index)
    plt.close()
    plt.clf()
    plt.cla()

    panel_sed_fig = vh.plot_cigale_sed_panel(cigale_flux_file_name='flux_file.dat',
                                     hdu_best_model_file_name=best_model_file,
                                     flux_fitted_model_file_name='out/results.fits',
                                     hst_file_name_list=hst_file_name_list, nircam_file_name_list=nircam_file_name_list,
                                     miri_file_name_list=miri_file_name_list,
                                     cutout_size=(2, 2), cutout_pos=cutout_pos, cutout_pos_list=position_list,
                                     hdu_number_hst=0, hdu_number_nircam='SCI', hdu_number_miri=0,
                                     filter_list=None, cigale_flux_file_col_name_ord=None, index_cigale_table=list_index,
                                     cigale_filt_names=None, cigale_logo_file_name=None, circle_rad=ap_list,
                                     filter_colors=None)

    panel_sed_fig.savefig('plot_output/sed_panel_%i.png' % cluster_index)
    panel_sed_fig.savefig('plot_output/sed_panel_%i.pdf' % cluster_index)
    plt.close()
    plt.clf()
    plt.cla()





