import numpy as np
# import phangs_cigale_fit
from astropy.io import fits

import pcigale

aperture_1 = 0.031
aperture_2 = 0.062
aperture_3 = 0.093
aperture_4 = 0.124
aperture_5 = 0.155
aperture_6 = 0.186

# get unit transformation factor
folder_hst = '/home/benutzer/data/PHANGS-HST/ngc7496/ngc7496_mosaics_01sep2020/'
file_name_hst_f275w = folder_hst + 'hlsp_phangs-hst_hst_wfc3-uvis_ngc7496_f275w_v1_exp-drc-sci.fits'
file_name_hst_f336w = folder_hst + 'hlsp_phangs-hst_hst_wfc3-uvis_ngc7496_f336w_v1_exp-drc-sci.fits'
file_name_hst_f438w = folder_hst + 'hlsp_phangs-hst_hst_wfc3-uvis_ngc7496_f438w_v1_exp-drc-sci.fits'
file_name_hst_f555w = folder_hst + 'hlsp_phangs-hst_hst_wfc3-uvis_ngc7496_f555w_v1_exp-drc-sci.fits'
file_name_hst_f814w = folder_hst + 'hlsp_phangs-hst_hst_wfc3-uvis_ngc7496_f814w_v1_exp-drc-sci.fits'

conv_fact_f275w = fits.open(file_name_hst_f275w)[0].header['PHOTFNU']
conv_fact_f336w = fits.open(file_name_hst_f336w)[0].header['PHOTFNU']
conv_fact_f438w = fits.open(file_name_hst_f438w)[0].header['PHOTFNU']
conv_fact_f555w = fits.open(file_name_hst_f555w)[0].header['PHOTFNU']
conv_fact_f814w = fits.open(file_name_hst_f814w)[0].header['PHOTFNU']


cluster_list = np.genfromtxt('data/candidates_embd_clus_v2p1.txt', dtype=object)
names = cluster_list[0]

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
flux_F770W = data[:, names == b'flux_F770W_6'] * 1e9
flux_F1000W = data[:, names == b'flux_F1000W_6'] * 1e9
flux_F1130W = data[:, names == b'flux_F1130W_6'] * 1e9
flux_F2100W = data[:, names == b'flux_F2100W_6'] * 1e9

flux_F275W_err = data[:, names == b'er_flux_F275W_5'] * conv_fact_f275w * 1e3
flux_F336W_err = data[:, names == b'er_flux_F336W_5'] * conv_fact_f336w * 1e3
flux_F438W_err = data[:, names == b'er_flux_F438W_5'] * conv_fact_f438w * 1e3
flux_F555W_err = data[:, names == b'er_flux_F555W_5'] * conv_fact_f555w * 1e3
flux_F814W_err = data[:, names == b'er_flux_F814W_5'] * conv_fact_f814w * 1e3
flux_F200W_err = data[:, names == b'er_flux_F200W_5'] * 1e9
flux_F300M_err = data[:, names == b'er_flux_F300M_5'] * 1e9
flux_F335M_err = data[:, names == b'er_flux_F335M_5'] * 1e9
flux_F360M_err = data[:, names == b'er_flux_F360M_5'] * 1e9
flux_F770W_err = data[:, names == b'er_flux_F770W_6'] * 1e9
flux_F1000W_err = data[:, names == b'er_flux_F1000W_6'] * 1e9
flux_F1130W_err = data[:, names == b'er_flux_F1130W_6'] * 1e9
flux_F2100W_err = data[:, names == b'er_flux_F2100W_6'] * 1e9

flux_list = np.array([flux_F275W, flux_F336W, flux_F438W, flux_F555W, flux_F814W,
                      flux_F200W, flux_F300M, flux_F335M, flux_F360M,
                      flux_F770W, flux_F1000W, flux_F1130W, flux_F2100W])
err_list = np.array([flux_F275W_err, flux_F336W_err, flux_F438W_err, flux_F555W_err, flux_F814W_err,
                     flux_F200W_err, flux_F300M_err, flux_F335M_err, flux_F360M_err,
                     flux_F770W_err, flux_F1000W_err, flux_F1130W_err, flux_F2100W_err])

mask_zero_flux = np.where(flux_list == 0)
flux_list[mask_zero_flux] = err_list[mask_zero_flux]

# change non detections to -1
mask_not_detected = np.where(flux_list < 3*err_list)
err_list[mask_not_detected] = -1


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

flux_file.writelines("# id             redshift  ")
for index in range(len(band_names)):
    flux_file.writelines(band_names[index] + "   ")
flux_file.writelines(" \n")


for cluster_index in range(flux_list.shape[1]):
    flux_file.writelines(" %i   0.0   " % cluster_index)
    for band_index in range(flux_list.shape[0]):
        flux_file.writelines("%.15f   " % flux_list[band_index, cluster_index])
        flux_file.writelines("%.15f   " % err_list[band_index, cluster_index])
    flux_file.writelines(" \n")



flux_file.close()
