import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

from astropy.io import fits
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from visualization_helper import VisualizeHelper as vh


# a = np.arange(0, 218, 6)
# b = ''
# for number in a:
#     b = b+str(number) + ', '
# print(b)
# exit()

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

cluster_list = np.genfromtxt('data/candidates_embd_clus_v2p1.txt', dtype=object)
names = cluster_list[0]
cluster_list[cluster_list == b'""'] = b'nan'
data = np.array(cluster_list[1:], dtype=float)

ra = data[:, names == b'raj2000']
dec = data[:, names == b'dej2000']
index_of_cluster_list = [53]

# possibles apertures in arc seconds
ap_list = [0.031, 0.062, 0.093, 0.124, 0.155, 0.186]
star_aperture_rad = ap_list[4]
ext_aperture_rad = ap_list[4]


hst_ap_list = np.array(vh.get_50p_aperture(instrument='hst'))
nircam_ap_list = np.array(vh.get_50p_aperture(instrument='nircam'))
miri_ap_list = np.array(vh.get_50p_aperture(instrument='miri'))


cluster_dict = {}
for list_index in index_of_cluster_list:

    cutout_pos = SkyCoord(ra=ra[list_index], dec=dec[list_index], unit=(u.degree, u.degree), frame='fk5')

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
                                                   hdu_number_hst=0, hdu_number_nircam='SCI', hdu_number_miri=0,
                                                   hdu_err_number_hst=0, hdu_err_number_nircam='ERR', hdu_err_number_miri=0,
                                                   flux_unit='mJy', re_center_peak=True)

    new_flux_list = np.array([flux_dict['flux_f275w'], flux_dict['flux_f336w'], flux_dict['flux_f438w'],
                              flux_dict['flux_f555w'], flux_dict['flux_f814w'],
                              flux_dict['flux_f200w'], flux_dict['flux_f300m'], flux_dict['flux_f335m'],
                              flux_dict['flux_f360m'],
                              flux_dict['flux_f770w'], flux_dict['flux_f1000w'], flux_dict['flux_f1130w'],
                              flux_dict['flux_f2100w']])


    new_flux_err_list = np.array([flux_dict['flux_err_f275w'], flux_dict['flux_err_f336w'], flux_dict['flux_err_f438w'],
                                  flux_dict['flux_err_f555w'], flux_dict['flux_err_f814w'],
                                  flux_dict['flux_err_f200w'], flux_dict['flux_err_f300m'], flux_dict['flux_err_f335m'],
                                  flux_dict['flux_err_f360m'],
                                  flux_dict['flux_err_f770w'], flux_dict['flux_err_f1000w'], flux_dict['flux_err_f1130w'],
                                  flux_dict['flux_err_f2100w']])

    upper_limit = (new_flux_list / new_flux_err_list < 5) | (new_flux_list < 0)
    print('upper_limit ', upper_limit)
    cluster_dict.update({'flux_%i' % list_index: new_flux_list,
                         'flux_err_%i' % list_index: new_flux_err_list,
                         'upper_limit_%i' % list_index: upper_limit})

    #
    # plt.scatter(np.array(wave_list), np.array(flux_list[:, list_index]))
    # plt.scatter(np.array(wave_list), np.array(new_flux_list))
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()


band_names = ('F275W_UVIS_CHIP2', 'F275W_UVIS_CHIP2_err', 'F336W_UVIS_CHIP2', 'F336W_UVIS_CHIP2_err',
              'F438W_UVIS_CHIP2', 'F438W_UVIS_CHIP2_err', 'F555W_UVIS_CHIP2', 'F555W_UVIS_CHIP2_err',
              'F814W_UVIS_CHIP2', 'F814W_UVIS_CHIP2_err', 'jwst.nircam.F200W', 'jwst.nircam.F200W_err',
              'jwst.nircam.F300M', 'jwst.nircam.F300M_err', 'jwst.nircam.F335M', 'jwst.nircam.F335M_err',
              'jwst.nircam.F360M', 'jwst.nircam.F360M_err', 'jwst.miri.F770W', 'jwst.miri.F770W_err',
              'jwst.miri.F1000W', 'jwst.miri.F1000W_err', 'jwst.miri.F1130W', 'jwst.miri.F1130W_err',
              'jwst.miri.F2100W', 'jwst.miri.F2100W_err')


band_string = 'bands = '
for index in range(len(band_names)):
    band_string += band_names[index]
    if index < (len(band_names) -1):
        band_string += ', '
print(band_string)

flux_file = open("flux_file.dat", "w")

flux_file.writelines("# id             redshift  distance   ")
for index in range(len(band_names)):
    flux_file.writelines(band_names[index] + "   ")
flux_file.writelines(" \n")


# for cluster_index in range(flux_list.shape[2]):
#     flux_file.writelines(" %i   0.0   " % cluster_index)
#     for band_index in range(flux_list.shape[0]):
#         flux_file.writelines("%.15f   " % flux_list[band_index, cluster_index])
#         flux_file.writelines("%.15f   " % err_list[band_index, cluster_index])
#     flux_file.writelines(" \n")


for cluster_index in index_of_cluster_list:
    flux_file.writelines(" %i   0.0   18.7  " % cluster_index)
    for band_index in range(len(new_flux_list)):
        if cluster_dict['upper_limit_%i' % cluster_index][band_index]:
            flux_file.writelines("%.15f   " % (np.max([cluster_dict['flux_err_%i' % cluster_index][band_index], 0]) +
                                               cluster_dict['flux_err_%i' % cluster_index][band_index]))
            flux_file.writelines("%.15f   " % (-cluster_dict['flux_err_%i' % cluster_index][band_index]))
        else:
            flux_file.writelines("%.15f   " % (cluster_dict['flux_%i' % cluster_index][band_index]))
            flux_file.writelines("%.15f   " % (cluster_dict['flux_err_%i' % cluster_index][band_index]))
    flux_file.writelines(" \n")

flux_file.close()

















