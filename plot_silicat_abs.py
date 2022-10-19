import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

from astropy.io import fits
from visualization_helper import VisualizeHelper as vh

wave_list = vh.get_wavelength_list_hst_nircam_miri()

cluster_list = np.genfromtxt('data/candidates_embd_clus_v2p1.txt', dtype=object)
names = cluster_list[0]
cluster_list[cluster_list == b'""'] = b'nan'
data = np.array(cluster_list[1:], dtype=float)
ra = data[:, names == b'raj2000']
dec = data[:, names == b'dej2000']

# estimate the upper limits
upper_lim_f275w = []
upper_lim_f336w = []
upper_lim_f438w = []
upper_lim_f555w = []
upper_lim_f814w = []
for cluster_index in range(len(ra)):
    result_dict = np.load('data_output/result_dict_%i.npy' % cluster_index, allow_pickle=True).item()
    flux_list = result_dict['flux_%i' % cluster_index]
    flux_err_list = result_dict['flux_err_%i' % cluster_index]
    upper_limit = result_dict['upper_limit_%i' % cluster_index].T[0]
    if upper_limit[0] | (flux_list[0]/flux_err_list[0]<5):
        upper_lim_f275w.append(flux_err_list[0])
    if upper_limit[1] | (flux_list[1]/flux_err_list[1]<5):
        upper_lim_f336w.append(flux_err_list[1])
    if upper_limit[2] | (flux_list[2]/flux_err_list[2]<5):
        upper_lim_f438w.append(flux_err_list[2])
    if upper_limit[3] | (flux_list[3]/flux_err_list[3]<5):
        upper_lim_f555w.append(flux_err_list[3])
    if upper_limit[4] | (flux_list[4] / flux_err_list[4] < 5):
        upper_lim_f814w.append(flux_err_list[4])

print(upper_lim_f275w)
print(upper_lim_f336w)
print(upper_lim_f438w)
print(upper_lim_f555w)
print(upper_lim_f814w)

print('upper_lim_f275w ', np.mean(upper_lim_f275w), ' mJy')
print('upper_lim_f336w ', np.mean(upper_lim_f336w), ' mJy')
print('upper_lim_f438w ', np.mean(upper_lim_f438w), ' mJy')
print('upper_lim_f555w ', np.mean(upper_lim_f555w), ' mJy')
print('upper_lim_f814w ', np.mean(upper_lim_f814w), ' mJy')

print('upper_lim_f275w ', -2.5 * np.log10(np.mean(upper_lim_f275w) * 1e-3) + 8.9, ' mag')
print('upper_lim_f336w ', -2.5 * np.log10(np.mean(upper_lim_f336w) * 1e-3) + 8.9, ' mag')
print('upper_lim_f438w ', -2.5 * np.log10(np.mean(upper_lim_f438w) * 1e-3) + 8.9, ' mag')
print('upper_lim_f555w ', -2.5 * np.log10(np.mean(upper_lim_f555w) * 1e-3) + 8.9, ' mag')
print('upper_lim_f814w ', -2.5 * np.log10(np.mean(upper_lim_f814w) * 1e-3) + 8.9, ' mag')

print('upper_lim_f275w ', np.mean(upper_lim_f275w)*5, ' mJy')
print('upper_lim_f336w ', np.mean(upper_lim_f336w)*5, ' mJy')
print('upper_lim_f438w ', np.mean(upper_lim_f438w)*5, ' mJy')
print('upper_lim_f555w ', np.mean(upper_lim_f555w)*5, ' mJy')
print('upper_lim_f814w ', np.mean(upper_lim_f814w)*5, ' mJy')

print('upper_lim_f275w ', -2.5 * np.log10(np.mean(upper_lim_f275w) * 5e-3) + 8.9, ' mag')
print('upper_lim_f336w ', -2.5 * np.log10(np.mean(upper_lim_f336w) * 5e-3) + 8.9, ' mag')
print('upper_lim_f438w ', -2.5 * np.log10(np.mean(upper_lim_f438w) * 5e-3) + 8.9, ' mag')
print('upper_lim_f555w ', -2.5 * np.log10(np.mean(upper_lim_f555w) * 5e-3) + 8.9, ' mag')
print('upper_lim_f814w ', -2.5 * np.log10(np.mean(upper_lim_f814w) * 5e-3) + 8.9, ' mag')


figure = plt.figure(figsize=(9, 6))
fontsize = 19
ax = figure.add_axes([0.105, 0.1, 0.89, 0.895])


for cluster_index in [4,7,8,36,37,43,48,50,53,59,62,63]:
    if cluster_index in [4,7,36]:
        linestyle = '--'
        color = 'g'
    else:
        linestyle = '-'
        color = 'r'

    result_dict = np.load('data_output/result_dict_%i.npy' % cluster_index, allow_pickle=True).item()
    flux_list = result_dict['flux_%i' % cluster_index]
    flux_err_list = result_dict['flux_err_%i' % cluster_index]
    upper_limit = result_dict['upper_limit_%i' % cluster_index].T[0]

    if cluster_index not in [4, 7, 36]:
        ax.scatter(wave_list[5:] * 1e-3, flux_list[5:], color='k')
        ax.plot(wave_list[5:] * 1e-3, flux_list[5:], linestyle=linestyle, color='k')
    else:
        ax.scatter(wave_list * 1e-3, flux_list, color='k')
        ax.plot(wave_list * 1e-3, flux_list, linestyle=linestyle, color='k')


upper_upper = [np.max(upper_lim_f275w), np.max(upper_lim_f336w), np.max(upper_lim_f438w), np.max(upper_lim_f555w), np.max(upper_lim_f814w)]
lower_upper = [np.min(upper_lim_f275w), np.min(upper_lim_f336w), np.min(upper_lim_f438w), np.min(upper_lim_f555w), np.min(upper_lim_f814w)]

ax.plot([], [], color='k', label='Embedded')
ax.plot([], [], color='k', linestyle='--', label='Intermediate')

ax.errorbar(wave_list[0]*1e-3, np.mean(upper_lim_f275w)*5, yerr=np.mean(upper_lim_f275w)*0.8,
            ecolor='grey', elinewidth=2, capsize=5, uplims=True, xlolims=False)
ax.errorbar(wave_list[1]*1e-3, np.mean(upper_lim_f336w)*5, yerr=np.mean(upper_lim_f336w)*0.8,
            ecolor='grey', elinewidth=2, capsize=5, uplims=True, xlolims=False)
ax.errorbar(wave_list[2]*1e-3, np.mean(upper_lim_f438w)*5, yerr=np.mean(upper_lim_f438w)*0.8,
            ecolor='grey', elinewidth=2, capsize=5, uplims=True, xlolims=False)
ax.errorbar(wave_list[3]*1e-3, np.mean(upper_lim_f555w)*5, yerr=np.mean(upper_lim_f555w)*0.8,
            ecolor='grey', elinewidth=2, capsize=5, uplims=True, xlolims=False)
ax.errorbar(wave_list[4]*1e-3, np.mean(upper_lim_f814w)*5, yerr=np.mean(upper_lim_f814w)*0.8,
            ecolor='grey', elinewidth=2, capsize=5, uplims=True, xlolims=False)

# ax.fill_between(wave_list[0:5], upper_upper, lower_upper, color='grey')

# ax.set_ylim(3e-6, 4e-1)

ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(frameon=False, ncols=1, fontsize=fontsize)
ax.set_xlabel('Wavelength [$\mu$m]', labelpad=-3.0, fontsize=fontsize)
ax.set_ylabel(r'F$_{\nu}$ [mJy]', labelpad=-3.0, fontsize=fontsize)
ax.tick_params(axis='both', which='both', width=2, direction='in', color='k', labelsize=fontsize)

# plt.show()
plt.savefig('plot_output/sed_comparison_black.png')
plt.savefig('plot_output/sed_comparison_black.pdf')
exit()

