import numpy as np
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
ra_img_ctr = ra_cluster_list[cluster_index] - 0.12/3600
dec_img_ctr = dec_cluster_list[cluster_index] + 0.09/3600
cutout_pos = SkyCoord(ra=ra_img_ctr, dec=dec_img_ctr, unit=(u.degree, u.degree), frame='fk5')
circ_pos = SkyCoord(ra=ra_cluster, dec=dec_cluster, unit=(u.degree, u.degree), frame='fk5')
cutout_size = (3, 3)

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

channel_hst_f275w = 'F275W'
channel_hst_f336w = 'F336W'
channel_hst_f438w = 'F438W'
channel_hst_f555w = 'F555W'
channel_hst_f814w = 'F814W'

channel_jwst_f200w = 'F200W'
channel_jwst_f300m = 'F300M'
channel_jwst_f335m = 'F335M'
channel_jwst_f360m = 'F360M'

channel_jwst_f770w = 'F770W'
channel_jwst_f1000w = 'F1000W'
channel_jwst_f1130w = 'F1130W'
channel_jwst_f2100w = 'F2100W'

hst_file_name_list = [file_name_hst_f275w, file_name_hst_f438w, file_name_hst_f555w, file_name_hst_f814w]
hst_channel_list = [channel_hst_f275w, channel_hst_f438w, channel_hst_f555w, channel_hst_f814w]
nircam_file_name_list = [file_name_jwst_f200w, file_name_jwst_f300m, file_name_jwst_f335m, file_name_jwst_f360m]
nircam_channel_list = [channel_jwst_f200w, channel_jwst_f300m, channel_jwst_f335m, channel_jwst_f360m]
miri_file_name_list = [file_name_jwst_f770w, file_name_jwst_f1000w, file_name_jwst_f1130w, file_name_jwst_f2100w]
miri_channel_list = [channel_jwst_f770w, channel_jwst_f1000w, channel_jwst_f1130w, channel_jwst_f2100w]

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

fig.savefig('plot_output/embedded_cluster_%i.png' % cluster_index)
fig.savefig('plot_output/embedded_cluster_%i.pdf' % cluster_index)
