import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

from visualization_helper import VisualizeHelper as vh


folder_hst = '/home/benutzer/data/PHANGS-HST/ngc7496/ngc7496_mosaics_01sep2020/'
folder_jwst = '/home/benutzer/data/PHANGS-JWST/ngc7496/'
folder_alma = '/home/benutzer/data/PHANGS-ALMA/ngc7496/'
folder_muse = '/home/benutzer/data/PHANGS-MUSE/ngc7496/'

file_name_hst_f275w = folder_hst + 'hlsp_phangs-hst_hst_wfc3-uvis_ngc7496_f275w_v1_exp-drc-sci.fits'
file_name_hst_f336w = folder_hst + 'hlsp_phangs-hst_hst_wfc3-uvis_ngc7496_f336w_v1_exp-drc-sci.fits'
file_name_hst_f438w = folder_hst + 'hlsp_phangs-hst_hst_wfc3-uvis_ngc7496_f438w_v1_exp-drc-sci.fits'
file_name_hst_f555w = folder_hst + 'hlsp_phangs-hst_hst_wfc3-uvis_ngc7496_f555w_v1_exp-drc-sci.fits'
file_name_hst_f814w = folder_hst + 'hlsp_phangs-hst_hst_wfc3-uvis_ngc7496_f814w_v1_exp-drc-sci.fits'

file_name_jwst_f200w = folder_jwst + 'ngc7496_nircam_lv3_f200w_i2d_align.fits'
file_name_jwst_f300m = folder_jwst + 'ngc7496_nircam_lv3_f300m_i2d_align.fits'
file_name_jwst_f335m = folder_jwst + 'ngc7496_nircam_lv3_f335m_i2d_align.fits'
file_name_jwst_f360m = folder_jwst + 'ngc7496_nircam_lv3_f360m_i2d_align.fits'

file_name_jwst_f770w = folder_jwst + 'ngc7496_miri_f770w_anchored.fits'
file_name_jwst_f1000w = folder_jwst + 'ngc7496_miri_f1000w_anchored.fits'
file_name_jwst_f1130w = folder_jwst + 'ngc7496_miri_f1130w_anchored.fits'
file_name_jwst_f2100w = folder_jwst + 'ngc7496_miri_f2100w_anchored.fits'

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
channel_list = [channel_hst_f275w, channel_hst_f336w, channel_hst_f438w, channel_hst_f555w, channel_hst_f814w,
                channel_jwst_f200w, channel_jwst_f300m, channel_jwst_f335m, channel_jwst_f360m,
                channel_jwst_f770w, channel_jwst_f1000w, channel_jwst_f1130w, channel_jwst_f2100w]

# get PSF
hst_psf_path = '/home/benutzer/data/PHANGS-HST/psf/'
# jwst_nircam_psf_path = '/home/benutzer/software/python_packages/make_convolution_kernel/psf/JWST/NIRCam/'
# jwst_miri_psf_path = '/home/benutzer/software/python_packages/make_convolution_kernel/psf/JWST/MIRI/'
jwst_psf_path = '/home/benutzer/data/PHANGS-JWST/psfs/'

psf_file_hst_f275w = hst_psf_path + 'PSFEFF_WFC3UV_F275W_C0.fits'
psf_file_hst_f336w = hst_psf_path + 'PSFEFF_WFC3UV_F336W_C0.fits'
psf_file_hst_f438w = hst_psf_path + 'PSFEFF_WFC3UV_F438W_C0.fits'
psf_file_hst_f555w = hst_psf_path + 'PSFEFF_WFC3UV_F555W_C0.fits'
psf_file_hst_f814w = hst_psf_path + 'PSFEFF_WFC3UV_F814W_C0.fits'
psf_file_jwst_f200w = jwst_psf_path + 'F200W.fits'
psf_file_jwst_f300m = jwst_psf_path + 'F300M.fits'
psf_file_jwst_f335m = jwst_psf_path + 'F335M.fits'
psf_file_jwst_f360m = jwst_psf_path + 'F360M.fits'
psf_file_jwst_f770w = jwst_psf_path + 'F770W.fits'
psf_file_jwst_f1000w = jwst_psf_path + 'F1000W.fits'
psf_file_jwst_f1130w = jwst_psf_path + 'F1130W.fits'
psf_file_jwst_f2100w = jwst_psf_path + 'F2100W.fits'



cluster_list = np.genfromtxt('data/candidates_embd_clus_v2p1.txt', skip_header=1)
# embedded clusters are: index 4,7,8,36,37,43,48,50,53,38,59,62,63


scaling_list = [1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.5, 1.5, 1.5,
                    3.0, 3.0, 3.0, 3.0]
circle_radius_list = [0.2, 0.2, 0.2, 0.2, 0.2,
                          0.2, 0.2, 0.2, 0.2,
                          0.8, 0.8, 0.8, 0.8]

for cluster_index in range(2):
    ra_cluster_list = cluster_list[:, 2]
    dec_cluster_list = cluster_list[:, 3]
    ra_cluster = ra_cluster_list[cluster_index]
    dec_cluster = dec_cluster_list[cluster_index]
    cutout_pos = SkyCoord(ra=ra_cluster, dec=dec_cluster, unit=(u.degree, u.degree), frame='fk5')
    cutout_size = (4.5, 4.5)


    cutout_dict = {}
    cutout_dict.update(vh.sep_source_extr(img_file=file_name_hst_f275w, cutout_pos=cutout_pos,
                                          cutout_size=cutout_size, obs='hst', hdu_number=0,
                                          psf_file_name=psf_file_hst_f275w, chan_name=channel_hst_f275w,
                                          deblend_nthresh=10, deblend_cont=0.0001, snr_thresh=1.0))
    cutout_dict.update(vh.sep_source_extr(img_file=file_name_hst_f336w, cutout_pos=cutout_pos,
                                          cutout_size=cutout_size, obs='hst', hdu_number=0,
                                          psf_file_name=psf_file_hst_f336w, chan_name=channel_hst_f336w,
                                          deblend_nthresh=10, deblend_cont=0.0001, snr_thresh=1.0))
    cutout_dict.update(vh.sep_source_extr(img_file=file_name_hst_f438w, cutout_pos=cutout_pos,
                                          cutout_size=cutout_size, obs='hst', hdu_number=0,
                                          psf_file_name=psf_file_hst_f438w, chan_name=channel_hst_f438w,
                                          deblend_nthresh=10, deblend_cont=0.0001, snr_thresh=1.0))
    cutout_dict.update(vh.sep_source_extr(img_file=file_name_hst_f555w, cutout_pos=cutout_pos,
                                          cutout_size=cutout_size, obs='hst', hdu_number=0,
                                          psf_file_name=psf_file_hst_f555w, chan_name=channel_hst_f555w,
                                          deblend_nthresh=10, deblend_cont=0.0001, snr_thresh=1.0))
    cutout_dict.update(vh.sep_source_extr(img_file=file_name_hst_f814w, cutout_pos=cutout_pos,
                                          cutout_size=cutout_size, obs='hst', hdu_number=0,
                                          psf_file_name=psf_file_hst_f814w, chan_name=channel_hst_f814w,
                                          deblend_nthresh=10, deblend_cont=0.0001, snr_thresh=1.0))

    cutout_dict.update(vh.sep_source_extr(img_file=file_name_jwst_f200w, cutout_pos=cutout_pos,
                                          cutout_size=cutout_size, obs='jwst', hdu_number='SCI',
                                          psf_file_name=psf_file_jwst_f200w, chan_name=channel_jwst_f200w,
                                          deblend_nthresh=10000, deblend_cont=0.000001, snr_thresh=3.0))
    cutout_dict.update(vh.sep_source_extr(img_file=file_name_jwst_f300m, cutout_pos=cutout_pos,
                                          cutout_size=cutout_size, obs='jwst', hdu_number='SCI',
                                          psf_file_name=psf_file_jwst_f300m, chan_name=channel_jwst_f300m,
                                          deblend_nthresh=10000, deblend_cont=0.000001, snr_thresh=1.0))
    cutout_dict.update(vh.sep_source_extr(img_file=file_name_jwst_f335m, cutout_pos=cutout_pos,
                                          cutout_size=cutout_size, obs='jwst', hdu_number='SCI',
                                          psf_file_name=psf_file_jwst_f335m, chan_name=channel_jwst_f335m,
                                          deblend_nthresh=10000, deblend_cont=0.000001, snr_thresh=1.0))
    cutout_dict.update(vh.sep_source_extr(img_file=file_name_jwst_f360m, cutout_pos=cutout_pos,
                                          cutout_size=cutout_size, obs='jwst', hdu_number='SCI',
                                          psf_file_name=psf_file_jwst_f360m, chan_name=channel_jwst_f360m,
                                          deblend_nthresh=10000, deblend_cont=0.000001, snr_thresh=1.0))

    cutout_dict.update(vh.sep_source_extr(img_file=file_name_jwst_f770w, cutout_pos=cutout_pos,
                                          cutout_size=cutout_size, obs='jwst', hdu_number=0,
                                          psf_file_name=psf_file_jwst_f770w, chan_name=channel_jwst_f770w,
                                          deblend_nthresh=10000, deblend_cont=0.000001, snr_thresh=0.5))
    cutout_dict.update(vh.sep_source_extr(img_file=file_name_jwst_f1000w, cutout_pos=cutout_pos,
                                          cutout_size=cutout_size, obs='jwst', hdu_number=0,
                                          psf_file_name=psf_file_jwst_f1000w, chan_name=channel_jwst_f1000w,
                                          deblend_nthresh=10000, deblend_cont=0.000001, snr_thresh=0.5))
    cutout_dict.update(vh.sep_source_extr(img_file=file_name_jwst_f1130w, cutout_pos=cutout_pos,
                                          cutout_size=cutout_size, obs='jwst', hdu_number=0,
                                          psf_file_name=psf_file_jwst_f1130w, chan_name=channel_jwst_f1130w,
                                          deblend_nthresh=10000, deblend_cont=0.000001, snr_thresh=0.5))
    cutout_dict.update(vh.sep_source_extr(img_file=file_name_jwst_f2100w, cutout_pos=cutout_pos,
                                          cutout_size=cutout_size, obs='jwst', hdu_number=0,
                                          psf_file_name=psf_file_jwst_f2100w, chan_name=channel_jwst_f2100w,
                                          deblend_nthresh=10000, deblend_cont=0.000001, snr_thresh=0.5))

    fig = vh.plot_sep_source_ext(cutout_dict=cutout_dict, channel_list=channel_list, circle_position=cutout_pos,
                                 circle_radius_list=circle_radius_list)

    fig.savefig('plot_output/source_identification_%i.png' % cluster_index)

    fig.clf()
    plt.close()

    # choose the fluxes !


    fig = vh.plot_sep_flux_ext(cutout_dict=cutout_dict, channel_list=channel_list,
                               circle_position=cutout_pos,
                               circle_radius_list=circle_radius_list,
                               scaling_list=scaling_list)
    fig.savefig('plot_output/flux_extraction_%i.png' % cluster_index)

    fig.clf()
    plt.close()

    flux_dict = vh.get_sep_source_fluxes(cutout_dict=cutout_dict, channel_list=channel_list,
                                         circle_position=cutout_pos,
                                         circle_radius_list=circle_radius_list,
                                         scaling_list=scaling_list)
    print(flux_dict)
    np.save('data/flux_dict_%i.npy' % cluster_index, flux_dict)






