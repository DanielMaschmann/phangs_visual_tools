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

file_name_jwst_f200w = folder_jwst + 'ngc1365_nircam_lv3_f200w_i2d_align.fits'
file_name_jwst_f300m = folder_jwst + 'ngc1365_nircam_lv3_f300m_i2d_align.fits'
file_name_jwst_f335m = folder_jwst + 'ngc1365_nircam_lv3_f335m_i2d_align.fits'
file_name_jwst_f360m = folder_jwst + 'ngc1365_nircam_lv3_f360m_i2d_align.fits'

file_name_jwst_f770w = folder_jwst + 'ngc1365_miri_f770w_anchored_fixedsatur_cutout.sci.dzliu.fits'
file_name_jwst_f1000w = folder_jwst + 'ngc1365_miri_f1000w_anchored_fixedsatur_cutout.sci.dzliu.fits'
file_name_jwst_f1130w = folder_jwst + 'ngc1365_miri_f1130w_anchored_fixedsatur_cutout.sci.dzliu.fits'
file_name_jwst_f2100w = folder_jwst + 'ngc1365_miri_f2100w_anchored_fixedsatur_cutout.sci.dzliu.fits'

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

conv_fact_f275w = fits.open(file_name_hst_f275w)[0].header['PHOTFNU']
conv_fact_f336w = fits.open(file_name_hst_f336w)[0].header['PHOTFNU']
conv_fact_f438w = fits.open(file_name_hst_f438w)[0].header['PHOTFNU']
conv_fact_f555w = fits.open(file_name_hst_f555w)[0].header['PHOTFNU']
conv_fact_f814w = fits.open(file_name_hst_f814w)[0].header['PHOTFNU']

hst_file_name_list = [file_name_hst_f275w, file_name_hst_f438w, file_name_hst_f555w, file_name_hst_f814w]
hst_channel_list = [channel_hst_f275w, channel_hst_f438w, channel_hst_f555w, channel_hst_f814w]
nircam_file_name_list = [file_name_jwst_f200w, file_name_jwst_f300m, file_name_jwst_f335m, file_name_jwst_f360m]
nircam_channel_list = [channel_jwst_f200w, channel_jwst_f300m, channel_jwst_f335m, channel_jwst_f360m]
miri_file_name_list = [file_name_jwst_f770w, file_name_jwst_f1000w, file_name_jwst_f1130w, file_name_jwst_f2100w]
miri_channel_list = [channel_jwst_f770w, channel_jwst_f1000w, channel_jwst_f1130w, channel_jwst_f2100w]



cluster_list = np.genfromtxt('data/ngc1365_for_daniel_sep_30_2022.txt', dtype=object)
names = cluster_list[0]
cluster_list[cluster_list == b'""'] = b'nan'
data = np.array(cluster_list[1:], dtype=float)

ra = data[:, names == b'raj2000']
dec = data[:, names == b'dej2000']

for index in range(len(ra)):
    cutout_pos = SkyCoord(ra=ra[index], dec=dec[index], unit=(u.degree, u.degree), frame='fk5')
    circ_pos = SkyCoord(ra=ra[index], dec=dec[index], unit=(u.degree, u.degree), frame='fk5')
    cutout_size = (3, 3)
    fig = vh.plot_multi_zoom_panel_hst_nircam_miri(hst_file_name_list=hst_file_name_list,
                                                   hst_channel_list=hst_channel_list,
                                                   nircam_file_name_list=nircam_file_name_list,
                                                   nircam_channel_list=nircam_channel_list,
                                                   miri_file_name_list=miri_file_name_list,
                                                   miri_channel_list=miri_channel_list,
                                                   cutout_pos=cutout_pos, cutout_size=cutout_size,
                                                   circ_pos=circ_pos, circ_rad=0.2,
                                                   hst_hdu_num=0, nircam_hdu_num='SCI', miri_hdu_num=0,
                                                   name_ra_offset=1.8, name_dec_offset=1.,
                                                   log_scale=True)
    fig.savefig('plot_output/cluster_%i.png' % index)
    fig.savefig('plot_output/cluster_%i.pdf' % index)
    fig.clf()
    plt.cla()
    plt.close()

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

flux_list = np.array([flux_F275W, flux_F336W, flux_F438W, flux_F555W, flux_F814W, flux_F200W, flux_F300M, flux_F335M, flux_F360M, flux_F770W, flux_F1000W, flux_F1130W, flux_F2100W])

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

# wavelength positions
wave_f275w = 0.275 * 1e3
wave_f336w = 0.336 * 1e3
wave_f438w = 0.438 * 1e3
wave_f555w = 0.555 * 1e3
wave_f814w = 0.814 * 1e3
wave_f200w = 2.00 * 1e3
wave_f300m = 3.00 * 1e3
wave_f335m = 3.35 * 1e3
wave_f360m = 3.60 * 1e3
wave_f770w = 7.70 * 1e3
wave_f1000w = 10.00 * 1e3
wave_f1130w = 11.30 * 1e3
wave_f2100w = 21.00 * 1e3

wave_list = [wave_f275w, wave_f336w, wave_f438w, wave_f555w, wave_f814w, wave_f200w, wave_f300m, wave_f335m, wave_f360m, wave_f770w, wave_f1000w, wave_f1130w, wave_f2100w]

color_list = ['tab:blue', 'tab:orange', 'tab:red',
              'tab:green', 'tab:pink', 'tab:brown']


for cluster_index in range(6):

    plt.scatter(wave_list, flux_list[:, cluster_index], color=color_list[cluster_index], label='%i' % cluster_index)
    plt.plot(wave_list, flux_list[:, cluster_index], color=color_list[cluster_index])

plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel('wavelength')
plt.ylabel('flux')
# plt.show()
plt.savefig('plot_output/sed_comparison.png')
plt.clf()
plt.cla()
plt.close()






