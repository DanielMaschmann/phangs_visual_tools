import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

from visualization_helper import VisualizeHelper as vh


cluster_list = np.genfromtxt('data/candidates_embd_clus_v2p1.txt', skip_header=1)
# embedded clusters are: index 4,7,8,36,37,43,48,50,53,38,59,62,63
cluster_index = 59
ra_cluster_list = cluster_list[:, 2]
dec_cluster_list = cluster_list[:, 3]
ra_cluster = ra_cluster_list[cluster_index]
dec_cluster = dec_cluster_list[cluster_index]
cutout_pos = SkyCoord(ra=ra_cluster, dec=dec_cluster, unit=(u.degree, u.degree), frame='fk5')
cutout_size = (2, 2)
rescale_fact = 1e6

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

file_name_jwst_f770w = folder_jwst + 'ngc7496_miri_f770w_anchored_at2100.fits'
file_name_jwst_f1000w = folder_jwst + 'ngc7496_miri_f1000w_anchored_at2100.fits'
file_name_jwst_f1130w = folder_jwst + 'ngc7496_miri_f1130w_anchored_at2100.fits'
file_name_jwst_f2100w = folder_jwst + 'ngc7496_miri_f2100w_anchored_at2100.fits'

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

# get fluxes for only one band



cutout_dict = {}
cutout_dict.update(vh.indi_point_fit2cutout(img_file=file_name_hst_f275w, cutout_pos=cutout_pos,
                                                 cutout_size=cutout_size, obs='hst', hdu_number=0,
                                                 psf_file_name=psf_file_hst_f275w, chan_name=channel_hst_f275w,
                                                 fwhm_point=0.3, thresh_point=4,
                                                 fitting_steps=50, fitting_rel_err=1e-16, rescaling_factor=rescale_fact))
cutout_dict.update(vh.indi_point_fit2cutout(img_file=file_name_hst_f336w, cutout_pos=cutout_pos,
                                                 cutout_size=cutout_size, obs='hst', hdu_number=0,
                                                 psf_file_name=psf_file_hst_f336w, chan_name=channel_hst_f336w,
                                                 fwhm_point=0.3, thresh_point=4,
                                                 fitting_steps=50, fitting_rel_err=1e-16, rescaling_factor=rescale_fact))
cutout_dict.update(vh.indi_point_fit2cutout(img_file=file_name_hst_f438w, cutout_pos=cutout_pos,
                                                 cutout_size=cutout_size, obs='hst', hdu_number=0,
                                                 psf_file_name=psf_file_hst_f438w, chan_name=channel_hst_f438w,
                                                 fwhm_point=0.3, thresh_point=4,
                                                 fitting_steps=50, fitting_rel_err=1e-16, rescaling_factor=rescale_fact))
cutout_dict.update(vh.indi_point_fit2cutout(img_file=file_name_hst_f555w, cutout_pos=cutout_pos,
                                                 cutout_size=cutout_size, obs='hst', hdu_number=0,
                                                 psf_file_name=psf_file_hst_f555w, chan_name=channel_hst_f555w,
                                                 fwhm_point=0.3, thresh_point=4,
                                                 fitting_steps=50, fitting_rel_err=1e-16, rescaling_factor=rescale_fact))
cutout_dict.update(vh.indi_point_fit2cutout(img_file=file_name_hst_f814w, cutout_pos=cutout_pos,
                                                 cutout_size=cutout_size, obs='hst', hdu_number=0,
                                                 psf_file_name=psf_file_hst_f814w, chan_name=channel_hst_f814w,
                                                 fwhm_point=0.3, thresh_point=4,
                                                 fitting_steps=50, fitting_rel_err=1e-16, rescaling_factor=rescale_fact))

cutout_dict.update(vh.indi_ext_fit2cutout(img_file=file_name_jwst_f200w, cutout_pos=cutout_pos,
                                                 cutout_size=cutout_size, obs='jwst', hdu_number='SCI',
                                                 psf_file_name=psf_file_jwst_f200w, chan_name=channel_jwst_f200w,
                                                 fwhm_point=0.3, thresh_point=4, box_size_extended_arc=0.1,
                                                 fitting_steps=50, fitting_rel_err=1e-16, rescaling_factor=rescale_fact))
cutout_dict.update(vh.indi_ext_fit2cutout(img_file=file_name_jwst_f300m, cutout_pos=cutout_pos,
                                                 cutout_size=cutout_size, obs='jwst', hdu_number='SCI',
                                                 psf_file_name=psf_file_jwst_f300m, chan_name=channel_jwst_f300m,
                                                 fwhm_point=0.3, thresh_point=4, box_size_extended_arc=0.1,
                                                 fitting_steps=50, fitting_rel_err=1e-16, rescaling_factor=rescale_fact))
cutout_dict.update(vh.indi_ext_fit2cutout(img_file=file_name_jwst_f335m, cutout_pos=cutout_pos,
                                                 cutout_size=cutout_size, obs='jwst', hdu_number='SCI',
                                                 psf_file_name=psf_file_jwst_f335m, chan_name=channel_jwst_f335m,
                                                 fwhm_point=0.3, thresh_point=4, box_size_extended_arc=0.2,
                                                 fitting_steps=50, fitting_rel_err=1e-16, rescaling_factor=rescale_fact))
cutout_dict.update(vh.indi_ext_fit2cutout(img_file=file_name_jwst_f360m, cutout_pos=cutout_pos,
                                                 cutout_size=cutout_size, obs='jwst', hdu_number='SCI',
                                                 psf_file_name=psf_file_jwst_f360m, chan_name=channel_jwst_f360m,
                                                 fwhm_point=0.3, thresh_point=4, box_size_extended_arc=0.2,
                                                 fitting_steps=50, fitting_rel_err=1e-16, rescaling_factor=rescale_fact))

cutout_dict.update(vh.indi_point_fit2cutout(img_file=file_name_jwst_f770w, cutout_pos=cutout_pos,
                                                 cutout_size=cutout_size, obs='jwst', hdu_number=0,
                                                 psf_file_name=psf_file_jwst_f770w, chan_name=channel_jwst_f770w,
                                                 fwhm_point=0.3, thresh_point=4,
                                                 fitting_steps=50, fitting_rel_err=1e-16, rescaling_factor=rescale_fact))
cutout_dict.update(vh.indi_point_fit2cutout(img_file=file_name_jwst_f1000w, cutout_pos=cutout_pos,
                                                 cutout_size=cutout_size, obs='jwst', hdu_number=0,
                                                 psf_file_name=psf_file_jwst_f1000w, chan_name=channel_jwst_f1000w,
                                                 fwhm_point=0.3, thresh_point=4,
                                                 fitting_steps=50, fitting_rel_err=1e-16, rescaling_factor=rescale_fact))
cutout_dict.update(vh.indi_point_fit2cutout(img_file=file_name_jwst_f1130w, cutout_pos=cutout_pos,
                                                 cutout_size=cutout_size, obs='jwst', hdu_number=0,
                                                 psf_file_name=psf_file_jwst_f1130w, chan_name=channel_jwst_f1130w,
                                                 fwhm_point=0.3, thresh_point=4,
                                                 fitting_steps=50, fitting_rel_err=1e-16, rescaling_factor=rescale_fact))
cutout_dict.update(vh.indi_point_fit2cutout(img_file=file_name_jwst_f2100w, cutout_pos=cutout_pos,
                                                 cutout_size=cutout_size, obs='jwst', hdu_number=0,
                                                 psf_file_name=psf_file_jwst_f2100w, chan_name=channel_jwst_f2100w,
                                                 fwhm_point=0.3, thresh_point=4,
                                                 fitting_steps=50, fitting_rel_err=1e-16, rescaling_factor=rescale_fact))


circle_rad_hst = 0.2
circle_rad_nircam = 0.1
circle_rad_miri = 0.5

# build up a figure
figure = plt.figure(figsize=(40, 18))

print(cutout_dict.keys())

ax_sources_model = figure.add_axes([-0.02, 0.68, 0.27, 0.27], projection=cutout_dict['obs_f200w'].wcs)
ax_sources_data = figure.add_axes([-0.02, 0.37, 0.27, 0.27], projection=cutout_dict['obs_f200w'].wcs)
ax_sources_residuals = figure.add_axes([-0.02, 0.05, 0.27, 0.27], projection=cutout_dict['obs_f200w'].wcs)

ax_model_f336w = figure.add_axes([0.14, 0.77, 0.18, 0.18], projection=cutout_dict['obs_f336w'].wcs)
ax_data_f336w = figure.add_axes([0.225, 0.77, 0.18, 0.18], projection=cutout_dict['obs_f336w'].wcs)
ax_residuals_f336w = figure.add_axes([0.31, 0.77, 0.18, 0.18], projection=cutout_dict['obs_f336w'].wcs)

ax_model_f438w = figure.add_axes([0.14, 0.525, 0.18, 0.18], projection=cutout_dict['obs_f438w'].wcs)
ax_data_f438w = figure.add_axes([0.225, 0.525, 0.18, 0.18], projection=cutout_dict['obs_f438w'].wcs)
ax_residuals_f438w = figure.add_axes([0.31, 0.525, 0.18, 0.18], projection=cutout_dict['obs_f438w'].wcs)

ax_model_f555w = figure.add_axes([0.14, 0.285, 0.18, 0.18], projection=cutout_dict['obs_f555w'].wcs)
ax_data_f555w = figure.add_axes([0.225, 0.285, 0.18, 0.18], projection=cutout_dict['obs_f555w'].wcs)
ax_residuals_f555w = figure.add_axes([0.31, 0.285, 0.18, 0.18], projection=cutout_dict['obs_f555w'].wcs)

ax_model_f814w = figure.add_axes([0.14, 0.05, 0.18, 0.18], projection=cutout_dict['obs_f814w'].wcs)
ax_data_f814w = figure.add_axes([0.225, 0.05, 0.18, 0.18], projection=cutout_dict['obs_f814w'].wcs)
ax_residuals_f814w = figure.add_axes([0.31, 0.05, 0.18, 0.18], projection=cutout_dict['obs_f814w'].wcs)


ax_model_f200w = figure.add_axes([0.41, 0.77, 0.18, 0.18], projection=cutout_dict['obs_f200w'].wcs)
ax_data_f200w = figure.add_axes([0.495, 0.77, 0.18, 0.18], projection=cutout_dict['obs_f200w'].wcs)
ax_residuals_f200w = figure.add_axes([0.58, 0.77, 0.18, 0.18], projection=cutout_dict['obs_f200w'].wcs)

ax_model_f300m = figure.add_axes([0.41, 0.525, 0.18, 0.18], projection=cutout_dict['obs_f300m'].wcs)
ax_data_f300m = figure.add_axes([0.495, 0.525, 0.18, 0.18], projection=cutout_dict['obs_f300m'].wcs)
ax_residuals_f300m = figure.add_axes([0.58, 0.525, 0.18, 0.18], projection=cutout_dict['obs_f300m'].wcs)

ax_model_f335m = figure.add_axes([0.41, 0.285, 0.18, 0.18], projection=cutout_dict['obs_f335m'].wcs)
ax_data_f335m = figure.add_axes([0.495, 0.285, 0.18, 0.18], projection=cutout_dict['obs_f335m'].wcs)
ax_residuals_f335m = figure.add_axes([0.58, 0.285, 0.18, 0.18], projection=cutout_dict['obs_f335m'].wcs)

ax_model_f360m = figure.add_axes([0.41, 0.05, 0.18, 0.18], projection=cutout_dict['obs_f360m'].wcs)
ax_data_f360m = figure.add_axes([0.495, 0.05, 0.18, 0.18], projection=cutout_dict['obs_f360m'].wcs)
ax_residuals_f360m = figure.add_axes([0.58, 0.05, 0.18, 0.18], projection=cutout_dict['obs_f360m'].wcs)


ax_model_f770w = figure.add_axes([0.68, 0.77, 0.18, 0.18], projection=cutout_dict['obs_f770w'].wcs)
ax_data_f770w = figure.add_axes([0.765, 0.77, 0.18, 0.18], projection=cutout_dict['obs_f770w'].wcs)
ax_residuals_f770w = figure.add_axes([0.85, 0.77, 0.18, 0.18], projection=cutout_dict['obs_f770w'].wcs)

ax_model_f1000w = figure.add_axes([0.68, 0.525, 0.18, 0.18], projection=cutout_dict['obs_f1000w'].wcs)
ax_data_f1000w = figure.add_axes([0.765, 0.525, 0.18, 0.18], projection=cutout_dict['obs_f1000w'].wcs)
ax_residuals_f1000w = figure.add_axes([0.85, 0.525, 0.18, 0.18], projection=cutout_dict['obs_f1000w'].wcs)

ax_model_f1130w = figure.add_axes([0.68, 0.285, 0.18, 0.18], projection=cutout_dict['obs_f1130w'].wcs)
ax_data_f1130w = figure.add_axes([0.765, 0.285, 0.18, 0.18], projection=cutout_dict['obs_f1130w'].wcs)
ax_residuals_f1130w = figure.add_axes([0.85, 0.285, 0.18, 0.18], projection=cutout_dict['obs_f1130w'].wcs)

ax_model_f2100w = figure.add_axes([0.68, 0.05, 0.18, 0.18], projection=cutout_dict['obs_f2100w'].wcs)
ax_data_f2100w = figure.add_axes([0.765, 0.05, 0.18, 0.18], projection=cutout_dict['obs_f2100w'].wcs)
ax_residuals_f2100w = figure.add_axes([0.85, 0.05, 0.18, 0.18], projection=cutout_dict['obs_f2100w'].wcs)

vh.plot_scarlet_results(ax_model=ax_sources_model, ax_data=ax_sources_data, ax_residuals=ax_sources_residuals,
                        obs=cutout_dict['obs_f200w'], model_frame=cutout_dict['model_frame_f200w'],
                        scarlet_sources=cutout_dict['scarlet_sources_f200w'], channel='f200w',
                        x_labels_model=False, x_labels_data=False, x_labels_residuals=True,
                        y_labels_model=True, y_labels_data=True, y_labels_residuals=True,
                        #ra_circle=ra_cluster, dec_circle=dec_cluster, circle_rad=0.1,
                        show_sources=True)

vh.plot_scarlet_results(ax_model=ax_model_f336w, ax_data=ax_data_f336w, ax_residuals=ax_residuals_f336w,
                        obs=cutout_dict['obs_f336w'], model_frame=cutout_dict['model_frame_f336w'],
                        scarlet_sources=cutout_dict['scarlet_sources_f336w'], channel='f336w',
                        x_labels_model=False, x_labels_data=False, x_labels_residuals=False,
                        y_labels_model=False, y_labels_data=False, y_labels_residuals=False,
                        ra_circle=ra_cluster, dec_circle=dec_cluster, circle_rad=circle_rad_hst)

vh.plot_scarlet_results(ax_model=ax_model_f438w, ax_data=ax_data_f438w, ax_residuals=ax_residuals_f438w,
                        obs=cutout_dict['obs_f438w'], model_frame=cutout_dict['model_frame_f438w'],
                        scarlet_sources=cutout_dict['scarlet_sources_f438w'], channel='f438w',
                        x_labels_model=False, x_labels_data=False, x_labels_residuals=False,
                        y_labels_model=False, y_labels_data=False, y_labels_residuals=False,
                        ra_circle=ra_cluster, dec_circle=dec_cluster, circle_rad=circle_rad_hst)

vh.plot_scarlet_results(ax_model=ax_model_f555w, ax_data=ax_data_f555w, ax_residuals=ax_residuals_f555w,
                        obs=cutout_dict['obs_f555w'], model_frame=cutout_dict['model_frame_f555w'],
                        scarlet_sources=cutout_dict['scarlet_sources_f555w'], channel='f555w',
                        x_labels_model=False, x_labels_data=False, x_labels_residuals=False,
                        y_labels_model=False, y_labels_data=False, y_labels_residuals=False,
                        ra_circle=ra_cluster, dec_circle=dec_cluster, circle_rad=circle_rad_hst)

vh.plot_scarlet_results(ax_model=ax_model_f814w, ax_data=ax_data_f814w, ax_residuals=ax_residuals_f814w,
                        obs=cutout_dict['obs_f814w'], model_frame=cutout_dict['model_frame_f814w'],
                        scarlet_sources=cutout_dict['scarlet_sources_f814w'], channel='f814w',
                        x_labels_model=False, x_labels_data=False, x_labels_residuals=False,
                        y_labels_model=False, y_labels_data=False, y_labels_residuals=False,
                        ra_circle=ra_cluster, dec_circle=dec_cluster, circle_rad=circle_rad_hst)


vh.plot_scarlet_results(ax_model=ax_model_f200w, ax_data=ax_data_f200w, ax_residuals=ax_residuals_f200w,
                        obs=cutout_dict['obs_f200w'], model_frame=cutout_dict['model_frame_f200w'],
                        scarlet_sources=cutout_dict['scarlet_sources_f200w'], channel='f200w',
                        x_labels_model=False, x_labels_data=False, x_labels_residuals=False,
                        y_labels_model=False, y_labels_data=False, y_labels_residuals=False,
                        ra_circle=ra_cluster, dec_circle=dec_cluster, circle_rad=circle_rad_nircam)

vh.plot_scarlet_results(ax_model=ax_model_f300m, ax_data=ax_data_f300m, ax_residuals=ax_residuals_f300m,
                        obs=cutout_dict['obs_f300m'], model_frame=cutout_dict['model_frame_f300m'],
                        scarlet_sources=cutout_dict['scarlet_sources_f300m'], channel='f300m',
                        x_labels_model=False, x_labels_data=False, x_labels_residuals=False,
                        y_labels_model=False, y_labels_data=False, y_labels_residuals=False,
                        ra_circle=ra_cluster, dec_circle=dec_cluster, circle_rad=circle_rad_nircam)

vh.plot_scarlet_results(ax_model=ax_model_f335m, ax_data=ax_data_f335m, ax_residuals=ax_residuals_f335m,
                        obs=cutout_dict['obs_f335m'], model_frame=cutout_dict['model_frame_f335m'],
                        scarlet_sources=cutout_dict['scarlet_sources_f335m'], channel='f335m',
                        x_labels_model=False, x_labels_data=False, x_labels_residuals=False,
                        y_labels_model=False, y_labels_data=False, y_labels_residuals=False,
                        ra_circle=ra_cluster, dec_circle=dec_cluster, circle_rad=circle_rad_nircam)

vh.plot_scarlet_results(ax_model=ax_model_f360m, ax_data=ax_data_f360m, ax_residuals=ax_residuals_f360m,
                        obs=cutout_dict['obs_f360m'], model_frame=cutout_dict['model_frame_f360m'],
                        scarlet_sources=cutout_dict['scarlet_sources_f360m'], channel='f360m',
                        x_labels_model=False, x_labels_data=False, x_labels_residuals=False,
                        y_labels_model=False, y_labels_data=False, y_labels_residuals=False,
                        ra_circle=ra_cluster, dec_circle=dec_cluster, circle_rad=circle_rad_nircam)


vh.plot_scarlet_results(ax_model=ax_model_f770w, ax_data=ax_data_f770w, ax_residuals=ax_residuals_f770w,
                        obs=cutout_dict['obs_f770w'], model_frame=cutout_dict['model_frame_f770w'],
                        scarlet_sources=cutout_dict['scarlet_sources_f770w'], channel='f770w',
                        x_labels_model=False, x_labels_data=False, x_labels_residuals=False,
                        y_labels_model=False, y_labels_data=False, y_labels_residuals=False,
                        ra_circle=ra_cluster, dec_circle=dec_cluster, circle_rad=circle_rad_miri)

vh.plot_scarlet_results(ax_model=ax_model_f1000w, ax_data=ax_data_f1000w, ax_residuals=ax_residuals_f1000w,
                        obs=cutout_dict['obs_f1000w'], model_frame=cutout_dict['model_frame_f1000w'],
                        scarlet_sources=cutout_dict['scarlet_sources_f1000w'], channel='f1000w',
                        x_labels_model=False, x_labels_data=False, x_labels_residuals=False,
                        y_labels_model=False, y_labels_data=False, y_labels_residuals=False,
                        ra_circle=ra_cluster, dec_circle=dec_cluster, circle_rad=circle_rad_miri)

vh.plot_scarlet_results(ax_model=ax_model_f1130w, ax_data=ax_data_f1130w, ax_residuals=ax_residuals_f1130w,
                        obs=cutout_dict['obs_f1130w'], model_frame=cutout_dict['model_frame_f1130w'],
                        scarlet_sources=cutout_dict['scarlet_sources_f1130w'], channel='f1130w',
                        x_labels_model=False, x_labels_data=False, x_labels_residuals=False,
                        y_labels_model=False, y_labels_data=False, y_labels_residuals=False,
                        ra_circle=ra_cluster, dec_circle=dec_cluster, circle_rad=circle_rad_miri)

vh.plot_scarlet_results(ax_model=ax_model_f2100w, ax_data=ax_data_f2100w, ax_residuals=ax_residuals_f2100w,
                        obs=cutout_dict['obs_f2100w'], model_frame=cutout_dict['model_frame_f2100w'],
                        scarlet_sources=cutout_dict['scarlet_sources_f2100w'], channel='f2100w',
                        x_labels_model=False, x_labels_data=False, x_labels_residuals=False,
                        y_labels_model=False, y_labels_data=False, y_labels_residuals=False,
                        ra_circle=ra_cluster, dec_circle=dec_cluster, circle_rad=circle_rad_miri)

plt.savefig('plot_output/cluster_fit.png')

figure.clf()
plt.clf()
plt.cla()


central_cluster_mask_f275w = np.sqrt((cutout_dict['ra_dec_point_src_f275w'][:, 0] - ra_cluster) ** 2 +
                                     (cutout_dict['ra_dec_point_src_f275w'][:, 1] - dec_cluster) ** 2) < circle_rad_hst/3600
print('central_cluster_mask_f275w ', sum(central_cluster_mask_f275w))
flux_f275w = vh.get_scarlet_src_flux(band='f275w', fit_result_dict=cutout_dict, mask=central_cluster_mask_f275w)

central_cluster_mask_f336w = np.sqrt((cutout_dict['ra_dec_point_src_f336w'][:, 0] - ra_cluster) ** 2 +
                                     (cutout_dict['ra_dec_point_src_f336w'][:, 1] - dec_cluster) ** 2) < circle_rad_hst/3600
print('central_cluster_mask_f336w ', sum(central_cluster_mask_f336w))
flux_f336w = vh.get_scarlet_src_flux(band='f336w', fit_result_dict=cutout_dict, mask=central_cluster_mask_f336w)

central_cluster_mask_f438w = np.sqrt((cutout_dict['ra_dec_point_src_f438w'][:, 0] - ra_cluster) ** 2 +
                                     (cutout_dict['ra_dec_point_src_f438w'][:, 1] - dec_cluster) ** 2) < circle_rad_hst/3600
print('central_cluster_mask_f438w ', sum(central_cluster_mask_f438w))
flux_f438w = vh.get_scarlet_src_flux(band='f438w', fit_result_dict=cutout_dict, mask=central_cluster_mask_f438w)

central_cluster_mask_f555w = np.sqrt((cutout_dict['ra_dec_point_src_f555w'][:, 0] - ra_cluster) ** 2 +
                                     (cutout_dict['ra_dec_point_src_f555w'][:, 1] - dec_cluster) ** 2) < circle_rad_hst/3600
print('central_cluster_mask_f555w ', sum(central_cluster_mask_f555w))
flux_f555w = vh.get_scarlet_src_flux(band='f555w', fit_result_dict=cutout_dict, mask=central_cluster_mask_f555w)

central_cluster_mask_f814w = np.sqrt((cutout_dict['ra_dec_point_src_f814w'][:, 0] - ra_cluster) ** 2 +
                                     (cutout_dict['ra_dec_point_src_f814w'][:, 1] - dec_cluster) ** 2) < circle_rad_hst/3600
print('central_cluster_mask_f814w ', sum(central_cluster_mask_f814w))
flux_f814w = vh.get_scarlet_src_flux(band='f814w', fit_result_dict=cutout_dict, mask=central_cluster_mask_f814w)


central_cluster_mask_f200w = np.sqrt((cutout_dict['ra_dec_point_src_f200w'][:, 0] - ra_cluster) ** 2 +
                                     (cutout_dict['ra_dec_point_src_f200w'][:, 1] - dec_cluster) ** 2) < circle_rad_nircam/3600
print('central_cluster_mask_f200w ', sum(central_cluster_mask_f200w))
flux_f200w = vh.get_scarlet_src_flux(band='f200w', fit_result_dict=cutout_dict, mask=central_cluster_mask_f200w)

central_cluster_mask_f300m = np.sqrt((cutout_dict['ra_dec_point_src_f300m'][:, 0] - ra_cluster) ** 2 +
                                     (cutout_dict['ra_dec_point_src_f300m'][:, 1] - dec_cluster) ** 2) < circle_rad_nircam/3600
print('central_cluster_mask_f300m ', sum(central_cluster_mask_f300m))
flux_f300m = vh.get_scarlet_src_flux(band='f300m', fit_result_dict=cutout_dict, mask=central_cluster_mask_f300m)

central_cluster_mask_f335m = np.sqrt((cutout_dict['ra_dec_point_src_f335m'][:, 0] - ra_cluster) ** 2 +
                                     (cutout_dict['ra_dec_point_src_f335m'][:, 1] - dec_cluster) ** 2) < circle_rad_nircam/3600
print('central_cluster_mask_f335m ', sum(central_cluster_mask_f335m))
flux_f335m = vh.get_scarlet_src_flux(band='f335m', fit_result_dict=cutout_dict, mask=central_cluster_mask_f335m)

central_cluster_mask_f360m = np.sqrt((cutout_dict['ra_dec_point_src_f360m'][:, 0] - ra_cluster) ** 2 +
                                     (cutout_dict['ra_dec_point_src_f360m'][:, 1] - dec_cluster) ** 2) < circle_rad_nircam/3600
print('central_cluster_mask_f360m ', sum(central_cluster_mask_f360m))
flux_f360m = vh.get_scarlet_src_flux(band='f360m', fit_result_dict=cutout_dict, mask=central_cluster_mask_f360m)

print(cutout_dict['ra_dec_point_src_f770w'])
central_cluster_mask_f770w = np.sqrt((cutout_dict['ra_dec_point_src_f770w'][0] - ra_cluster) ** 2 +
                                     (cutout_dict['ra_dec_point_src_f770w'][1] - dec_cluster) ** 2) < circle_rad_miri/3600
print('central_cluster_mask_f770w ', central_cluster_mask_f770w)
flux_f770w = vh.get_scarlet_src_flux(band='f770w', fit_result_dict=cutout_dict, mask=[central_cluster_mask_f770w])

central_cluster_mask_f1000w = np.sqrt((cutout_dict['ra_dec_point_src_f1000w'][0] - ra_cluster) ** 2 +
                                     (cutout_dict['ra_dec_point_src_f1000w'][1] - dec_cluster) ** 2) < circle_rad_miri/3600
print('central_cluster_mask_f1000w ', central_cluster_mask_f1000w)
flux_f1000w = vh.get_scarlet_src_flux(band='f1000w', fit_result_dict=cutout_dict, mask=[central_cluster_mask_f1000w])

central_cluster_mask_f1130w = np.sqrt((cutout_dict['ra_dec_point_src_f1130w'][:, 0] - ra_cluster) ** 2 +
                                     (cutout_dict['ra_dec_point_src_f1130w'][:, 1] - dec_cluster) ** 2) < circle_rad_miri/3600
print('central_cluster_mask_f1130w ', central_cluster_mask_f1130w)
flux_f1130w = vh.get_scarlet_src_flux(band='f1130w', fit_result_dict=cutout_dict, mask=central_cluster_mask_f1130w)

central_cluster_mask_f2100w = np.sqrt((cutout_dict['ra_dec_point_src_f2100w'][0] - ra_cluster) ** 2 +
                                     (cutout_dict['ra_dec_point_src_f2100w'][1] - dec_cluster) ** 2) < circle_rad_miri/3600
print('central_cluster_mask_f2100w ', central_cluster_mask_f2100w)
flux_f2100w = vh.get_scarlet_src_flux(band='f2100w', fit_result_dict=cutout_dict, mask=[central_cluster_mask_f2100w])


flux_f275w_err = flux_f275w * 0.1
flux_f336w_err = flux_f336w * 0.1
flux_f438w_err = flux_f438w * 0.1
flux_f555w_err = flux_f555w * 0.1
flux_f814w_err = flux_f814w * 0.1

flux_f200w_err = flux_f200w * 0.1
flux_f300m_err = flux_f300m * 0.1
flux_f335m_err = flux_f335m * 0.1
flux_f360m_err = flux_f360m * 0.1

flux_f770w_err = flux_f770w * 0.1
flux_f1000w_err = flux_f1000w * 0.1
flux_f1130w_err = flux_f1130w * 0.1
flux_f2100w_err = flux_f2100w * 0.1


print('flux_f275w ', flux_f275w, ' +/- ', flux_f275w_err)
print('flux_f336w ', flux_f336w, ' +/- ', flux_f336w_err)
print('flux_f438w ', flux_f438w, ' +/- ', flux_f438w_err)
print('flux_f555w ', flux_f555w, ' +/- ', flux_f555w_err)
print('flux_f814w ', flux_f814w, ' +/- ', flux_f814w_err)
print('flux_f200w ', flux_f200w, ' +/- ', flux_f200w_err)
print('flux_f300m ', flux_f300m, ' +/- ', flux_f300m_err)
print('flux_f335m ', flux_f335m, ' +/- ', flux_f335m_err)
print('flux_f360m ', flux_f360m, ' +/- ', flux_f360m_err)
print('flux_f770w ', flux_f770w, ' +/- ', flux_f770w_err)
print('flux_f1000w ', flux_f1000w, ' +/- ', flux_f1000w_err)
print('flux_f1130w ', flux_f1130w, ' +/- ', flux_f1130w_err)
print('flux_f2100w ', flux_f2100w, ' +/- ', flux_f2100w_err)

flux_dict = {
    'flux_f275w': flux_f275w/rescale_fact, 'flux_f336w': flux_f336w/rescale_fact, 'flux_f438w': flux_f438w/rescale_fact, 'flux_f555w': flux_f555w/rescale_fact,
    'flux_f814w': flux_f814w/rescale_fact,
    'flux_f275w_err': flux_f275w_err/rescale_fact, 'flux_f336w_err': flux_f336w_err/rescale_fact, 'flux_f438w_err': flux_f438w_err/rescale_fact, 'flux_f555w_err': flux_f555w_err/rescale_fact,
    'flux_f814w_err': flux_f814w_err/rescale_fact,
    'flux_f200w': flux_f200w/rescale_fact, 'flux_f300m': flux_f300m/rescale_fact, 'flux_f335m': flux_f335m/rescale_fact, 'flux_f360m': flux_f360m/rescale_fact,
    'flux_f770w': flux_f770w/rescale_fact, 'flux_f1000w': flux_f1000w/rescale_fact, 'flux_f1130w': flux_f1130w/rescale_fact, 'flux_f2100w': flux_f2100w/rescale_fact,
    'flux_f200w_err': flux_f200w_err/rescale_fact, 'flux_f300m_err': flux_f300m_err/rescale_fact, 'flux_f335m_err': flux_f335m_err/rescale_fact, 'flux_f360m_err': flux_f360m_err/rescale_fact,
    'flux_f770w_err': flux_f770w_err/rescale_fact, 'flux_f1000w_err': flux_f1000w_err/rescale_fact, 'flux_f1130w_err': flux_f1130w_err/rescale_fact, 'flux_f2100w_err': flux_f2100w_err/rescale_fact
}

np.save('data/flux_dict.npy', flux_dict)

figure = plt.figure(figsize=(16, 7))

ax = figure.add_axes([0.08, 0.1, 0.9, 0.89])

ax.scatter(0.275, flux_f275w, s=80)
ax.scatter(0.336, flux_f336w, s=80)
ax.scatter(0.438, flux_f438w, s=80)
ax.scatter(0.555, flux_f555w, s=80)
ax.scatter(0.814, flux_f814w, s=80)

ax.scatter(2.0, flux_f200w, s=80)
ax.scatter(3.0, flux_f300m, s=80)
ax.scatter(3.35, flux_f335m, s=80)
ax.scatter(3.60, flux_f360m, s=80)

ax.scatter(7.7, flux_f770w, s=80)
ax.scatter(10.0, flux_f1000w, s=80)
ax.scatter(11.3, flux_f1130w, s=80)
ax.scatter(21.0, flux_f2100w, s=80)

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlabel(r'Wavelength, $\mu$ m', fontsize=20)
ax.set_ylabel(r'Flux, mJy', fontsize=20)

ax.tick_params(axis='both', which='both', width=1.5, direction='in', color='k', labelsize=20)

# plt.show()
plt.savefig('plot_output/sed_flux_only_%i.png' % cluster_index)


