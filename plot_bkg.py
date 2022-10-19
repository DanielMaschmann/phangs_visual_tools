import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

from astropy.io import fits
from visualization_helper import VisualizeHelper as vh

# a = np.arange(0, 218, 6)
# b = ''
# for number in a:
#     b = b+str(number) + ', '
# print(b)
# exit()


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

wave_list = vh.get_wavelength_list_hst_nircam_miri()

# possibles apertures in arc seconds
ap_list = [0.031, 0.062, 0.093, 0.124, 0.155, 0.186]
star_aperture_rad = ap_list[4]
ext_aperture_rad = ap_list[4]

hst_ap_list = np.array(vh.get_50p_aperture(instrument='hst'))
nircam_ap_list = np.array(vh.get_50p_aperture(instrument='nircam'))
miri_ap_list = np.array(vh.get_50p_aperture(instrument='miri'))


cluster_dict = {}
list_index = 2

cutout_pos = SkyCoord(ra=ra[list_index], dec=dec[list_index], unit=(u.degree, u.degree), frame='fk5')

flux_dict = vh.get_multiband_flux_from_circ_ap(cutout_pos=cutout_pos,
                                               hst_list=hst_file_name_list, hst_err_list=hst_err_files,
                                               hst_ap_rad_list=hst_ap_list,
                                               nircam_list=nircam_file_name_list, nircam_ap_rad_list=nircam_ap_list,
                                               miri_list=miri_file_name_list, miri_err_list=miri_err_file_name_list,
                                               miri_ap_rad_list=miri_ap_list,
                                               hst_channel_list=hst_channel_list, nircam_channel_list=nircam_channel_list,
                                               miri_channel_list=miri_channel_list,
                                               hdu_number_hst=0, hdu_number_nircam='SCI', hdu_number_miri=0,
                                               hdu_err_number_hst=0, hdu_err_number_nircam='ERR', hdu_err_number_miri=0,
                                               flux_unit='mJy')


fig = plt.figure(figsize=(30, 12))
fontsize = 33

ax_image_f275w = fig.add_axes([0.03, 0.73, 0.13, 0.13], projection=flux_dict['flux_cutout_f275w'].wcs)
ax_image_f336w = fig.add_axes([0.09, 0.73, 0.13, 0.13], projection=flux_dict['flux_cutout_f336w'].wcs)
ax_image_f438w = fig.add_axes([0.15, 0.73, 0.13, 0.13], projection=flux_dict['flux_cutout_f438w'].wcs)
ax_image_f555w = fig.add_axes([0.21, 0.73, 0.13, 0.13], projection=flux_dict['flux_cutout_f555w'].wcs)
ax_image_f814w = fig.add_axes([0.27, 0.73, 0.13, 0.13], projection=flux_dict['flux_cutout_f814w'].wcs)
ax_hst_img_cbar = fig.add_axes([0.11, 0.92, 0.20, 0.01])


ax_image_f200w = fig.add_axes([0.36, 0.73, 0.13, 0.13], projection=flux_dict['flux_cutout_f200w'].wcs)
ax_image_f300m = fig.add_axes([0.44, 0.73, 0.13, 0.13], projection=flux_dict['flux_cutout_f300m'].wcs)
ax_image_f335m = fig.add_axes([0.5, 0.73, 0.13, 0.13], projection=flux_dict['flux_cutout_f335m'].wcs)
ax_image_f360m = fig.add_axes([0.56, 0.73, 0.13, 0.13], projection=flux_dict['flux_cutout_f360m'].wcs)
ax_nircam_img_cbar = fig.add_axes([0.44, 0.92, 0.20, 0.01])

ax_image_f770w = fig.add_axes([0.65, 0.73, 0.13, 0.13], projection=flux_dict['flux_cutout_f770w'].wcs)
ax_image_f1000w = fig.add_axes([0.71, 0.73, 0.13, 0.13], projection=flux_dict['flux_cutout_f1000w'].wcs)
ax_image_f1130w = fig.add_axes([0.77, 0.73, 0.13, 0.13], projection=flux_dict['flux_cutout_f1130w'].wcs)
ax_image_f2100w = fig.add_axes([0.86, 0.73, 0.13, 0.13], projection=flux_dict['flux_cutout_f2100w'].wcs)
ax_miri_img_cbar = fig.add_axes([0.71, 0.92, 0.20, 0.01])

ax_err_f275w = fig.add_axes([0.03, 0.36, 0.13, 0.13], projection=flux_dict['flux_err_cutout_f275w'].wcs)
ax_err_f336w = fig.add_axes([0.09, 0.36, 0.13, 0.13], projection=flux_dict['flux_err_cutout_f336w'].wcs)
ax_err_f438w = fig.add_axes([0.15, 0.36, 0.13, 0.13], projection=flux_dict['flux_err_cutout_f438w'].wcs)
ax_err_f555w = fig.add_axes([0.21, 0.36, 0.13, 0.13], projection=flux_dict['flux_err_cutout_f555w'].wcs)
ax_err_f814w = fig.add_axes([0.27, 0.36, 0.13, 0.13], projection=flux_dict['flux_err_cutout_f814w'].wcs)
ax_hst_err_cbar = fig.add_axes([0.11, 0.57, 0.20, 0.01])


ax_err_f200w = fig.add_axes([0.36, 0.36, 0.13, 0.13], projection=flux_dict['flux_err_cutout_f200w'].wcs)
ax_err_f300m = fig.add_axes([0.44, 0.36, 0.13, 0.13], projection=flux_dict['flux_err_cutout_f300m'].wcs)
ax_err_f335m = fig.add_axes([0.5, 0.36, 0.13, 0.13], projection=flux_dict['flux_err_cutout_f335m'].wcs)
ax_err_f360m = fig.add_axes([0.56, 0.36, 0.13, 0.13], projection=flux_dict['flux_err_cutout_f360m'].wcs)
ax_nircam_err_cbar = fig.add_axes([0.44, 0.57, 0.20, 0.01])


ax_err_f770w = fig.add_axes([0.65, 0.36, 0.13, 0.13], projection=flux_dict['flux_err_cutout_f770w'].wcs)
ax_err_f1000w = fig.add_axes([0.71, 0.36, 0.13, 0.13], projection=flux_dict['flux_err_cutout_f1000w'].wcs)
ax_err_f1130w = fig.add_axes([0.77, 0.36, 0.13, 0.13], projection=flux_dict['flux_err_cutout_f1130w'].wcs)
ax_err_f2100w = fig.add_axes([0.86, 0.36, 0.13, 0.13], projection=flux_dict['flux_err_cutout_f2100w'].wcs)
ax_miri_err_cbar = fig.add_axes([0.71, 0.57, 0.20, 0.01])

ax_bkg_f275w = fig.add_axes([0.03, 0.03, 0.13, 0.13])
ax_bkg_f336w = fig.add_axes([0.09, 0.03, 0.13, 0.13])
ax_bkg_f438w = fig.add_axes([0.15, 0.03, 0.13, 0.13])
ax_bkg_f555w = fig.add_axes([0.21, 0.03, 0.13, 0.13])
ax_bkg_f814w = fig.add_axes([0.27, 0.03, 0.13, 0.13])
ax_bkg_f200w = fig.add_axes([0.36, 0.03, 0.13, 0.13])
ax_bkg_f300m = fig.add_axes([0.44, 0.03, 0.13, 0.13])
ax_bkg_f335m = fig.add_axes([0.5, 0.03, 0.13, 0.13])
ax_bkg_f360m = fig.add_axes([0.56, 0.03, 0.13, 0.13])
ax_bkg_f770w = fig.add_axes([0.65, 0.03, 0.13, 0.13])
ax_bkg_f1000w = fig.add_axes([0.71, 0.03, 0.13, 0.13])
ax_bkg_f1130w = fig.add_axes([0.77, 0.03, 0.13, 0.13])
ax_bkg_f2100w = fig.add_axes([0.86, 0.03, 0.13, 0.13])


cbar_hst_img = ax_image_f275w.imshow(flux_dict['flux_cutout_f275w'].data, vmin=-0.3*1e-4, vmax=3.9*1e-4)
ax_image_f336w.imshow(flux_dict['flux_cutout_f336w'].data, vmin=-0.3*1e-4, vmax=3.9*1e-4)
ax_image_f438w.imshow(flux_dict['flux_cutout_f438w'].data, vmin=-0.3*1e-4, vmax=3.9*1e-4)
ax_image_f555w.imshow(flux_dict['flux_cutout_f555w'].data, vmin=-0.3*1e-4, vmax=3.9*1e-4)
ax_image_f814w.imshow(flux_dict['flux_cutout_f814w'].data, vmin=-0.3*1e-4, vmax=3.9*1e-4)
fig.colorbar(cbar_hst_img, cax=ax_hst_img_cbar, orientation='horizontal')
# ax_color_bar_2.set_xlabel(r'log(S /[MJy / sr])', labelpad=5, fontsize=fontsize)
ax_hst_img_cbar.set_xlabel(r'Surface brightness mJy', labelpad=5, fontsize=fontsize-10)
ax_hst_img_cbar.xaxis.set_label_position('top')
ax_hst_img_cbar.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False, labeltop=True, labelsize=fontsize-10)

cbar_nircam_img = ax_image_f200w.imshow(flux_dict['flux_cutout_f200w'].data, vmin=-0.3*1e-3, vmax=9*1e-3)
ax_image_f300m.imshow(flux_dict['flux_cutout_f300m'].data, vmin=-0.3*1e-3, vmax=9*1e-3)
ax_image_f335m.imshow(flux_dict['flux_cutout_f335m'].data, vmin=-0.3*1e-3, vmax=9*1e-3)
ax_image_f360m.imshow(flux_dict['flux_cutout_f360m'].data, vmin=-0.3*1e-3, vmax=9*1e-3)
fig.colorbar(cbar_nircam_img, cax=ax_nircam_img_cbar, orientation='horizontal')
# ax_color_bar_2.set_xlabel(r'log(S /[MJy / sr])', labelpad=5, fontsize=fontsize)
ax_nircam_img_cbar.set_xlabel(r'Surface brightness mJy', labelpad=5, fontsize=fontsize-10)
ax_nircam_img_cbar.xaxis.set_label_position('top')
ax_nircam_img_cbar.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False, labeltop=True, labelsize=fontsize-10)

cbar_miri_img = ax_image_f770w.imshow(flux_dict['flux_cutout_f770w'].data, vmin=-0.3*1e-1, vmax=3.9*1e-1)
ax_image_f1000w.imshow(flux_dict['flux_cutout_f1000w'].data, vmin=-0.3*1e-1, vmax=3.9*1e-1)
ax_image_f1130w.imshow(flux_dict['flux_cutout_f1130w'].data, vmin=-0.3*1e-1, vmax=3.9*1e-1)
ax_image_f2100w.imshow(flux_dict['flux_cutout_f2100w'].data, vmin=-0.3*1e-1, vmax=3.9*1e-1)
fig.colorbar(cbar_miri_img, cax=ax_miri_img_cbar, orientation='horizontal')
# ax_color_bar_2.set_xlabel(r'log(S /[MJy / sr])', labelpad=5, fontsize=fontsize)
ax_miri_img_cbar.set_xlabel(r'Surface brightness mJy', labelpad=5, fontsize=fontsize-10)
ax_miri_img_cbar.xaxis.set_label_position('top')
ax_miri_img_cbar.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False, labeltop=True, labelsize=fontsize-10)

ax_image_f275w.set_title('%.6f' % flux_dict['flux_f275w'], fontsize=fontsize-10)
ax_image_f336w.set_title('%.6f' % flux_dict['flux_f336w'], fontsize=fontsize-10)
ax_image_f438w.set_title('%.6f' % flux_dict['flux_f438w'], fontsize=fontsize-10)
ax_image_f555w.set_title('%.6f' % flux_dict['flux_f555w'], fontsize=fontsize-10)
ax_image_f814w.set_title('%.6f' % flux_dict['flux_f814w'], fontsize=fontsize-10)
ax_image_f200w.set_title('%.6f' % flux_dict['flux_f200w'], fontsize=fontsize-10)
ax_image_f300m.set_title('%.4f' % flux_dict['flux_f300m'], fontsize=fontsize-10)
ax_image_f335m.set_title('%.4f' % flux_dict['flux_f335m'], fontsize=fontsize-10)
ax_image_f360m.set_title('%.4f' % flux_dict['flux_f360m'], fontsize=fontsize-10)
ax_image_f770w.set_title('%.4f' % flux_dict['flux_f770w'], fontsize=fontsize-10)
ax_image_f1000w.set_title('%.4f' % flux_dict['flux_f1000w'], fontsize=fontsize-10)
ax_image_f1130w.set_title('%.4f' % flux_dict['flux_f1130w'], fontsize=fontsize-10)
ax_image_f2100w.set_title('%.4f' % flux_dict['flux_f2100w'], fontsize=fontsize-10)



cbar_hst_err = ax_err_f275w.imshow(flux_dict['flux_err_cutout_f275w'].data,  vmin=-0.3*1e-6, vmax=15*1e-6)
ax_err_f336w.imshow(flux_dict['flux_err_cutout_f336w'].data,  vmin=-0.3*1e-6, vmax=15*1e-6)
ax_err_f438w.imshow(flux_dict['flux_err_cutout_f438w'].data,  vmin=-0.3*1e-6, vmax=15*1e-6)
ax_err_f555w.imshow(flux_dict['flux_err_cutout_f555w'].data,  vmin=-0.3*1e-6, vmax=15*1e-6)
ax_err_f814w.imshow(flux_dict['flux_err_cutout_f814w'].data,  vmin=-0.3*1e-6, vmax=15*1e-6)
fig.colorbar(cbar_hst_err, cax=ax_hst_err_cbar, orientation='horizontal')
# ax_color_bar_2.set_xlabel(r'log(S /[MJy / sr])', labelpad=5, fontsize=fontsize)
ax_hst_err_cbar.set_xlabel(r'err Surface brightness mJy', labelpad=5, fontsize=fontsize-10)
ax_hst_err_cbar.xaxis.set_label_position('top')
ax_hst_err_cbar.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False, labeltop=True, labelsize=fontsize-10)
ax_hst_err_cbar.xaxis.get_offset_text().set_fontsize(fontsize-10)

cbar_nircam_err = ax_err_f200w.imshow(flux_dict['flux_err_cutout_f200w'].data, vmin=-0.3*1e-5, vmax=5*1e-5)
ax_err_f300m.imshow(flux_dict['flux_err_cutout_f300m'].data, vmin=-0.3*1e-5, vmax=5*1e-5)
ax_err_f335m.imshow(flux_dict['flux_err_cutout_f335m'].data, vmin=-0.3*1e-5, vmax=5*1e-5)
ax_err_f360m.imshow(flux_dict['flux_err_cutout_f360m'].data, vmin=-0.3*1e-5, vmax=5*1e-5)
fig.colorbar(cbar_nircam_err, cax=ax_nircam_err_cbar, orientation='horizontal')
# ax_color_bar_2.set_xlabel(r'log(S /[MJy / sr])', labelpad=5, fontsize=fontsize)
ax_nircam_err_cbar.set_xlabel(r'err Surface brightness mJy', labelpad=5, fontsize=fontsize-10)
ax_nircam_err_cbar.xaxis.set_label_position('top')
ax_nircam_err_cbar.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False, labeltop=True, labelsize=fontsize-10)
ax_nircam_err_cbar.xaxis.get_offset_text().set_fontsize(fontsize-10)

cbar_miri_err = ax_err_f770w.imshow(flux_dict['flux_err_cutout_f770w'].data, vmin=-0.3*1e-4, vmax=3*1e-4)
ax_err_f1000w.imshow(flux_dict['flux_err_cutout_f1000w'].data, vmin=-0.3*1e-4, vmax=3*1e-4)
ax_err_f1130w.imshow(flux_dict['flux_err_cutout_f1130w'].data, vmin=-0.3*1e-4, vmax=3*1e-4)
ax_err_f2100w.imshow(flux_dict['flux_err_cutout_f2100w'].data, vmin=-0.3*1e-4, vmax=3*1e-4)
fig.colorbar(cbar_miri_err, cax=ax_miri_err_cbar, orientation='horizontal')
# ax_color_bar_2.set_xlabel(r'log(S /[MJy / sr])', labelpad=5, fontsize=fontsize)
ax_miri_err_cbar.set_xlabel(r'err Surface brightness mJy', labelpad=5, fontsize=fontsize-10)
ax_miri_err_cbar.xaxis.set_label_position('top')
ax_miri_err_cbar.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False, labeltop=True, labelsize=fontsize-10)
ax_miri_err_cbar.xaxis.get_offset_text().set_fontsize(fontsize-10)

ax_err_f275w.set_title('%.6f' % flux_dict['flux_err_f275w'], fontsize=fontsize-10)
ax_err_f336w.set_title('%.6f' % flux_dict['flux_err_f336w'], fontsize=fontsize-10)
ax_err_f438w.set_title('%.6f' % flux_dict['flux_err_f438w'], fontsize=fontsize-10)
ax_err_f555w.set_title('%.6f' % flux_dict['flux_err_f555w'], fontsize=fontsize-10)
ax_err_f814w.set_title('%.6f' % flux_dict['flux_err_f814w'], fontsize=fontsize-10)
ax_err_f200w.set_title('%.6f' % flux_dict['flux_err_f200w'], fontsize=fontsize-10)
ax_err_f300m.set_title('%.4f' % flux_dict['flux_err_f300m'], fontsize=fontsize-10)
ax_err_f335m.set_title('%.4f' % flux_dict['flux_err_f335m'], fontsize=fontsize-10)
ax_err_f360m.set_title('%.4f' % flux_dict['flux_err_f360m'], fontsize=fontsize-10)
ax_err_f770w.set_title('%.4f' % flux_dict['flux_err_f770w'], fontsize=fontsize-10)
ax_err_f1000w.set_title('%.4f' % flux_dict['flux_err_f1000w'], fontsize=fontsize-10)
ax_err_f1130w.set_title('%.4f' % flux_dict['flux_err_f1130w'], fontsize=fontsize-10)
ax_err_f2100w.set_title('%.4f' % flux_dict['flux_err_f2100w'], fontsize=fontsize-10)



vh.plot_coord_circle(ax=ax_image_f275w, position=flux_dict['position_f275w'], radius=0.110*u.arcsec, color='r')
vh.plot_coord_circle(ax=ax_image_f336w, position=flux_dict['position_f336w'], radius=0.066*u.arcsec, color='r')
vh.plot_coord_circle(ax=ax_image_f438w, position=flux_dict['position_f438w'], radius=0.065*u.arcsec, color='r')
vh.plot_coord_circle(ax=ax_image_f555w, position=flux_dict['position_f555w'], radius=0.068*u.arcsec, color='r')
vh.plot_coord_circle(ax=ax_image_f814w, position=flux_dict['position_f814w'], radius=0.097*u.arcsec, color='r')
vh.plot_coord_circle(ax=ax_image_f200w, position=flux_dict['position_f200w'], radius=0.049*u.arcsec, color='r')
vh.plot_coord_circle(ax=ax_image_f300m, position=flux_dict['position_f300m'], radius=0.066*u.arcsec, color='r')
vh.plot_coord_circle(ax=ax_image_f335m, position=flux_dict['position_f335m'], radius=0.073*u.arcsec, color='r')
vh.plot_coord_circle(ax=ax_image_f360m, position=flux_dict['position_f360m'], radius=0.077*u.arcsec, color='r')
vh.plot_coord_circle(ax=ax_image_f770w, position=flux_dict['position_f770w'], radius=0.168*u.arcsec, color='r')
vh.plot_coord_circle(ax=ax_image_f1000w, position=flux_dict['position_f1000w'], radius=0.209*u.arcsec, color='r')
vh.plot_coord_circle(ax=ax_image_f1130w, position=flux_dict['position_f1130w'], radius=0.236*u.arcsec, color='r')
vh.plot_coord_circle(ax=ax_image_f2100w, position=flux_dict['position_f2100w'], radius=0.420*u.arcsec, color='r')







ax_bkg_f275w.set_title('%.6f' % (np.mean(flux_dict['bkg_f275w'].back())), fontsize=fontsize-10    )
ax_bkg_f336w.set_title('%.6f' % (np.mean(flux_dict['bkg_f336w'].back())), fontsize=fontsize-10    )
ax_bkg_f438w.set_title('%.6f' % (np.mean(flux_dict['bkg_f438w'].back())), fontsize=fontsize-10    )
ax_bkg_f555w.set_title('%.6f' % (np.mean(flux_dict['bkg_f555w'].back())), fontsize=fontsize-10    )
ax_bkg_f814w.set_title('%.6f' % (np.mean(flux_dict['bkg_f814w'].back())), fontsize=fontsize-10    )
ax_bkg_f200w.set_title('%.6f' % (np.mean(flux_dict['bkg_f200w'].back())), fontsize=fontsize-10    )
ax_bkg_f300m.set_title('%.6f' % (np.mean(flux_dict['bkg_f300m'].back())), fontsize=fontsize-10    )
ax_bkg_f335m.set_title('%.6f' % (np.mean(flux_dict['bkg_f335m'].back())), fontsize=fontsize-10    )
ax_bkg_f360m.set_title('%.6f' % (np.mean(flux_dict['bkg_f360m'].back())), fontsize=fontsize-10    )
ax_bkg_f770w.set_title('%.6f' % (np.mean(flux_dict['bkg_f770w'].back())), fontsize=fontsize-10    )
ax_bkg_f1000w.set_title('%.6f' % (np.mean(flux_dict['bkg_f1000w'].back())), fontsize=fontsize-10  )
ax_bkg_f1130w.set_title('%.6f' % (np.mean(flux_dict['bkg_f1130w'].back())), fontsize=fontsize-10  )
ax_bkg_f2100w.set_title('%.6f' % (np.mean(flux_dict['bkg_f2100w'].back())), fontsize=fontsize-10  )

ax_bkg_f275w.imshow(flux_dict['bkg_f275w'].back())
ax_bkg_f336w.imshow(flux_dict['bkg_f336w'].back())
ax_bkg_f438w.imshow(flux_dict['bkg_f438w'].back())
ax_bkg_f555w.imshow(flux_dict['bkg_f555w'].back())
ax_bkg_f814w.imshow(flux_dict['bkg_f814w'].back())
ax_bkg_f200w.imshow(flux_dict['bkg_f200w'].back())
ax_bkg_f300m.imshow(flux_dict['bkg_f300m'].back())
ax_bkg_f335m.imshow(flux_dict['bkg_f335m'].back())
ax_bkg_f360m.imshow(flux_dict['bkg_f360m'].back())
ax_bkg_f770w.imshow(flux_dict['bkg_f770w'].back())
ax_bkg_f1000w.imshow(flux_dict['bkg_f1000w'].back())
ax_bkg_f1130w.imshow(flux_dict['bkg_f1130w'].back())
ax_bkg_f2100w.imshow(flux_dict['bkg_f2100w'].back())


vh.erase_axis(ax=ax_image_f275w)
vh.erase_axis(ax=ax_image_f336w)
vh.erase_axis(ax=ax_image_f438w)
vh.erase_axis(ax=ax_image_f555w)
vh.erase_axis(ax=ax_image_f814w)
vh.erase_axis(ax=ax_image_f200w)
vh.erase_axis(ax=ax_image_f300m)
vh.erase_axis(ax=ax_image_f335m)
vh.erase_axis(ax=ax_image_f360m)
vh.erase_axis(ax=ax_image_f770w)
vh.erase_axis(ax=ax_image_f1000w)
vh.erase_axis(ax=ax_image_f1130w)
vh.erase_axis(ax=ax_image_f2100w)
vh.erase_axis(ax=ax_err_f275w)
vh.erase_axis(ax=ax_err_f336w)
vh.erase_axis(ax=ax_err_f438w)
vh.erase_axis(ax=ax_err_f555w)
vh.erase_axis(ax=ax_err_f814w)
vh.erase_axis(ax=ax_err_f200w)
vh.erase_axis(ax=ax_err_f300m)
vh.erase_axis(ax=ax_err_f335m)
vh.erase_axis(ax=ax_err_f360m)
vh.erase_axis(ax=ax_err_f770w)
vh.erase_axis(ax=ax_err_f1000w)
vh.erase_axis(ax=ax_err_f1130w)
vh.erase_axis(ax=ax_err_f2100w)
ax_bkg_f275w.axis('off')
ax_bkg_f336w.axis('off')
ax_bkg_f438w.axis('off')
ax_bkg_f555w.axis('off')
ax_bkg_f814w.axis('off')
ax_bkg_f200w.axis('off')
ax_bkg_f300m.axis('off')
ax_bkg_f335m.axis('off')
ax_bkg_f360m.axis('off')
ax_bkg_f770w.axis('off')
ax_bkg_f1000w.axis('off')
ax_bkg_f1130w.axis('off')
ax_bkg_f2100w.axis('off')


plt.savefig('plot_output/cutout_prop.png')
