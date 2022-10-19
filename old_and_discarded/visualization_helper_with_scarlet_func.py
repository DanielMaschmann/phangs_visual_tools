import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c as speed_of_light
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse

import astropy.units as u
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

from photutils.detection import DAOStarFinder

import sep

try:
    import scarlet
except ImportError:
    print('no scarlet package available ')


class VisualizeHelper:

    @staticmethod
    def get_img(file_name, hdu_number=0):
        """function to open hdu using astropy.

        Parameters
        ----------
        file_name : str
            file name to open
        hdu_number : int or str
            hdu number which should be opened. can be also a string such as 'SCI' for JWST images

        Returns
        -------
        2D array and astropy.wcs object
            returns the image in form of an array and the according WCS
        """
        # get hdu
        hdu = fits.open(file_name)
        # get header
        header = hdu[hdu_number].header
        # get WCS
        wcs = WCS(header)
        # update the header
        header.update(wcs.to_header())
        # reload the WCS and header
        header = hdu[hdu_number].header
        wcs = WCS(header)
        # load data
        data = hdu[hdu_number].data
        # PHOTFNU
        return data, header, wcs

    @staticmethod
    def get_img_cutout(img, wcs, coord, cutout_size):
        """function to cut out a region of a larger image with an WCS and perform a peak finder.
        Parameters
        ----------
        img : ndarray
            (Ny, Nx) image
        wcs : astropy.wcs.WCS()
            astropy world coordinate system object describing the parameter image
        coord : astropy.coordinates.SkyCoord
            astropy coordinate object to point to the selected area which to cutout
        cutout_size : float or tuple
            Units in arcsec. Cutout size of a box cutout. If float it will be used for both box length.

        Returns
        -------
        astropy.nddata.Cutout2D object
            cutout object of the initial image
        """

        if isinstance(cutout_size, tuple):
            size = cutout_size * u.arcsec
        elif isinstance(cutout_size, float) | isinstance(cutout_size, int):
            size = (cutout_size, cutout_size) * u.arcsec
        else:
            raise KeyError('cutout_size must be float or tuple')

        # check if cutout is inside the image
        pix_pos = wcs.world_to_pixel(coord)
        if (pix_pos[0] > 0) & (pix_pos[0] < img.shape[1]) & (pix_pos[1] > 0) & (pix_pos[1] < img.shape[0]):
            return Cutout2D(data=img, position=coord, size=size, wcs=wcs)
        else:
            cut_out = type('', (), {})()
            cut_out.data = None
            cut_out.wcs = None
            return cut_out

    @staticmethod
    def get_hst_cutout_from_file(file_name, hdu_number, cutout_pos, cutout_size, rescaling='Jy'):

        data, header, wcs = VisualizeHelper.get_img(file_name=file_name, hdu_number=hdu_number)

        if 'PHOTFNU' in header:
            conversion_factor = header['PHOTFNU']
        elif 'PHOTFLAM' in header:
            # wavelength in angstrom
            pivot_wavelength = header['PHOTPLAM']
            # inverse sensitivity, ergs/cm2/Ang/electron
            sensitivity = header['PHOTFLAM']
            # speed of light in Angstrom/s
            c = speed_of_light * 1e10
            # change the conversion facto to get erg s−1 cm−2 Hz−1
            f_nu = sensitivity * pivot_wavelength**2 / c
            # change to get Jy
            conversion_factor = f_nu * 1e23

        else:
            raise KeyError('there is no PHOTFNU or PHOTFLAM in the header')

        if rescaling == 'Jy':
            # rescale to Jy
            data *= conversion_factor
        elif rescaling == 'mJy':
            # rescale to Jy
            data *= conversion_factor * 1e3
        elif rescaling == 'MJy/sr':
            # rescale to Jy
            data *= conversion_factor
            # get the size of one pixel in sr with the factor 1e6 for the MJy later
            sr_per_square_deg = 0.00030461741978671
            pixel_area_size = wcs.proj_plane_pixel_area() * sr_per_square_deg * 1e6
            # change to MJy/sr
            data /= pixel_area_size.value
        else:
            raise KeyError('rescaling ', rescaling, ' not understand')

        return VisualizeHelper.get_img_cutout(img=data, wcs=wcs, coord=cutout_pos, cutout_size=cutout_size)

    @staticmethod
    def get_jwst_cutout_from_file(file_name, hdu_number, cutout_pos, cutout_size, rescaling='Jy'):

        # data is in MJy / sr
        data, header, wcs = VisualizeHelper.get_img(file_name=file_name, hdu_number=hdu_number)

        if rescaling == 'Jy':
            # rescale to Jy
            sr_per_square_deg = 0.00030461741978671
            pixel_area_size = wcs.proj_plane_pixel_area() * sr_per_square_deg
            data *= pixel_area_size.value
            # now put it to Jy
            data *= 1e6
        elif rescaling == 'mJy':
            # rescale to Jy
            sr_per_square_deg = 0.00030461741978671
            pixel_area_size = wcs.proj_plane_pixel_area() * sr_per_square_deg
            data *= pixel_area_size.value
            # now put it to Jy
            data *= 1e9
        elif rescaling == 'MJy/sr':
            pass
        else:
            raise KeyError('rescaling ', rescaling, ' not understand')

        return VisualizeHelper.get_img_cutout(img=data, wcs=wcs, coord=cutout_pos, cutout_size=cutout_size)

    @staticmethod
    def set_lim2cutout(ax, cutout, cutout_pos, edge_cut_ratio=100):

        coord_left_low = cutout.wcs.pixel_to_world(0, 0)
        coord_upper_right = cutout.wcs.pixel_to_world(cutout.data.shape[0], cutout.data.shape[1])

        pos_cut_diff_ra = (coord_left_low.ra - coord_upper_right.ra)
        pos_cut_diff_dec = (coord_upper_right.dec - coord_left_low.dec)

        lim_top_left = cutout.wcs.world_to_pixel(SkyCoord(cutout_pos.ra + pos_cut_diff_ra/2 - pos_cut_diff_ra/edge_cut_ratio,
                                                          cutout_pos.dec + pos_cut_diff_dec/2 - pos_cut_diff_dec/edge_cut_ratio))

        lim_bottom_right = cutout.wcs.world_to_pixel(SkyCoord(cutout_pos.ra - pos_cut_diff_ra/2 + pos_cut_diff_ra/edge_cut_ratio,
                                                              cutout_pos.dec - pos_cut_diff_dec/2 + pos_cut_diff_dec/edge_cut_ratio))

        ax.set_xlim(lim_top_left[0], lim_bottom_right[0])
        ax.set_ylim(lim_bottom_right[1], lim_top_left[1])

    @staticmethod
    def plot_multi_zoom_panel_hst_nircam_miri(hst_file_name_list, hst_channel_list,
                                              nircam_file_name_list, nircam_channel_list,
                                              miri_file_name_list, miri_channel_list,
                                              cutout_pos, cutout_size,
                                              circ_pos=False, circ_rad=0.5,
                                              hst_hdu_num=0, nircam_hdu_num='SCI', miri_hdu_num='SCI',
                                              fontsize=20, figsize=(18, 13),
                                              vmax_hst=None, vmax_nircam=None, vmax_miri=None,
                                              ticks_hst=None, ticks_nircam=None, ticks_miri=None,
                                              cmap_hst='Blues', cmap_nircam='Greens', cmap_miri='Reds',
                                              log_scale=False,
                                              name_ra_offset=2.4, name_dec_offset=1.5,
                                              ra_tick_num=3, dec_tick_num=3):

        cutout_hst_f275w = VisualizeHelper.get_hst_cutout_from_file(file_name=hst_file_name_list[0], hdu_number=hst_hdu_num,
                                                                    cutout_pos=cutout_pos, cutout_size=cutout_size,
                                                                    rescaling='MJy/sr')
        cutout_hst_f438w = VisualizeHelper.get_hst_cutout_from_file(file_name=hst_file_name_list[1], hdu_number=hst_hdu_num,
                                                                    cutout_pos=cutout_pos, cutout_size=cutout_size,
                                                                    rescaling='MJy/sr')
        cutout_hst_f555w = VisualizeHelper.get_hst_cutout_from_file(file_name=hst_file_name_list[2], hdu_number=hst_hdu_num,
                                                                    cutout_pos=cutout_pos, cutout_size=cutout_size,
                                                                    rescaling='MJy/sr')
        cutout_hst_f814w = VisualizeHelper.get_hst_cutout_from_file(file_name=hst_file_name_list[3], hdu_number=hst_hdu_num,
                                                                    cutout_pos=cutout_pos, cutout_size=cutout_size,
                                                                    rescaling='MJy/sr')

        cutout_jwst_f200w = VisualizeHelper.get_jwst_cutout_from_file(file_name=nircam_file_name_list[0],
                                                                      hdu_number=nircam_hdu_num, cutout_pos=cutout_pos,
                                                                      cutout_size=cutout_size, rescaling='MJy/sr')
        cutout_jwst_f300m = VisualizeHelper.get_jwst_cutout_from_file(file_name=nircam_file_name_list[1],
                                                                      hdu_number=nircam_hdu_num, cutout_pos=cutout_pos,
                                                                      cutout_size=cutout_size, rescaling='MJy/sr')
        cutout_jwst_f335m = VisualizeHelper.get_jwst_cutout_from_file(file_name=nircam_file_name_list[2],
                                                                      hdu_number=nircam_hdu_num, cutout_pos=cutout_pos,
                                                                      cutout_size=cutout_size, rescaling='MJy/sr')
        cutout_jwst_f360m = VisualizeHelper.get_jwst_cutout_from_file(file_name=nircam_file_name_list[3],
                                                                      hdu_number=nircam_hdu_num, cutout_pos=cutout_pos,
                                                                      cutout_size=cutout_size, rescaling='MJy/sr')
        cutout_jwst_f770w = VisualizeHelper.get_jwst_cutout_from_file(file_name=miri_file_name_list[0],
                                                                      hdu_number=miri_hdu_num, cutout_pos=cutout_pos,
                                                                      cutout_size=cutout_size, rescaling='MJy/sr')
        cutout_jwst_f1000w = VisualizeHelper.get_jwst_cutout_from_file(file_name=miri_file_name_list[1],
                                                                       hdu_number=miri_hdu_num, cutout_pos=cutout_pos,
                                                                       cutout_size=cutout_size, rescaling='MJy/sr')
        cutout_jwst_f1130w = VisualizeHelper.get_jwst_cutout_from_file(file_name=miri_file_name_list[2],
                                                                       hdu_number=miri_hdu_num, cutout_pos=cutout_pos,
                                                                       cutout_size=cutout_size, rescaling='MJy/sr')
        cutout_jwst_f2100w = VisualizeHelper.get_jwst_cutout_from_file(file_name=miri_file_name_list[3],
                                                                       hdu_number=miri_hdu_num, cutout_pos=cutout_pos,
                                                                       cutout_size=cutout_size, rescaling='MJy/sr')

        if vmax_hst is None:
            vmax_hst = np.nanmax([np.max(cutout_hst_f275w.data), np.max(cutout_hst_f438w.data),
                               np.max(cutout_hst_f555w.data), np.max(cutout_hst_f814w.data)])
        if vmax_nircam is None:
            vmax_nircam = np.nanmax([np.max(cutout_jwst_f200w.data), np.max(cutout_jwst_f300m.data),
                                  np.max(cutout_jwst_f335m.data), np.max(cutout_jwst_f360m.data)])
        if vmax_miri is None:
            vmax_miri = np.nanmax([np.max(cutout_jwst_f770w.data), np.max(cutout_jwst_f1000w.data),
                                np.max(cutout_jwst_f1130w.data), np.max(cutout_jwst_f2100w.data)])

        # build up a figure
        figure = plt.figure(figsize=figsize)


        ax_hst_f275w = figure.add_axes([0.0, 0.67, 0.3, 0.3], projection=cutout_hst_f275w.wcs)
        ax_hst_f438w = figure.add_axes([0.22, 0.67, 0.3, 0.3], projection=cutout_hst_f438w.wcs)
        ax_hst_f555w = figure.add_axes([0.44, 0.67, 0.3, 0.3], projection=cutout_hst_f555w.wcs)
        ax_hst_f814w = figure.add_axes([0.66, 0.67, 0.3, 0.3], projection=cutout_hst_f814w.wcs)
        ax_color_bar_hst = figure.add_axes([0.925, 0.68, 0.015, 0.28])

        ax_jwst_f200w = figure.add_axes([0.0, 0.365, 0.3, 0.3], projection=cutout_jwst_f200w.wcs)
        ax_jwst_f300m = figure.add_axes([0.22, 0.365, 0.3, 0.3], projection=cutout_jwst_f300m.wcs)
        ax_jwst_f335m = figure.add_axes([0.44, 0.365, 0.3, 0.3], projection=cutout_jwst_f335m.wcs)
        ax_jwst_f360m = figure.add_axes([0.66, 0.365, 0.3, 0.3], projection=cutout_jwst_f360m.wcs)
        ax_color_bar_nircam = figure.add_axes([0.925, 0.375, 0.015, 0.28])

        ax_jwst_f770w = figure.add_axes([0.0, 0.06, 0.3, 0.3], projection=cutout_jwst_f770w.wcs)
        ax_jwst_f1000w = figure.add_axes([0.22, 0.06, 0.3, 0.3], projection=cutout_jwst_f1000w.wcs)
        ax_jwst_f1130w = figure.add_axes([0.44, 0.06, 0.3, 0.3], projection=cutout_jwst_f1130w.wcs)
        ax_jwst_f2100w = figure.add_axes([0.66, 0.06, 0.3, 0.3], projection=cutout_jwst_f2100w.wcs)
        ax_color_bar_miri = figure.add_axes([0.925, 0.07, 0.015, 0.28])

        if log_scale:
            cb_hst = ax_hst_f275w.imshow(np.log10(cutout_hst_f275w.data), vmin=np.log10(vmax_hst/500),
                                         vmax=np.log10(vmax_hst/3), cmap=cmap_hst)
            ax_hst_f438w.imshow(np.log10(cutout_hst_f438w.data), vmin=np.log10(vmax_hst/500), vmax=np.log10(vmax_hst/3),
                                cmap=cmap_hst)
            ax_hst_f555w.imshow(np.log10(cutout_hst_f555w.data), vmin=np.log10(vmax_hst/500), vmax=np.log10(vmax_hst/3),
                                cmap=cmap_hst)
            ax_hst_f814w.imshow(np.log10(cutout_hst_f814w.data), vmin=np.log10(vmax_hst/500), vmax=np.log10(vmax_hst/3),
                                cmap=cmap_hst)

            cb_nircam = ax_jwst_f200w.imshow(np.log10(cutout_jwst_f200w.data), vmin=np.log10(vmax_nircam/500),
                                             vmax=np.log10(vmax_nircam/3), cmap=cmap_nircam)
            ax_jwst_f300m.imshow(np.log10(cutout_jwst_f300m.data), vmin=np.log10(vmax_nircam/500),
                                 vmax=np.log10(vmax_nircam/3), cmap=cmap_nircam)
            ax_jwst_f335m.imshow(np.log10(cutout_jwst_f335m.data), vmin=np.log10(vmax_nircam/500),
                                 vmax=np.log10(vmax_nircam/3), cmap=cmap_nircam)
            ax_jwst_f360m.imshow(np.log10(cutout_jwst_f360m.data), vmin=np.log10(vmax_nircam/500),
                                 vmax=np.log10(vmax_nircam/3), cmap=cmap_nircam)

            cb_miri = ax_jwst_f770w.imshow(np.log10(cutout_jwst_f770w.data), vmin=np.log10(vmax_miri/500),
                                           vmax=np.log10(vmax_miri/3), cmap=cmap_miri)
            ax_jwst_f1000w.imshow(np.log10(cutout_jwst_f1000w.data), vmin=np.log10(vmax_miri/500),
                                  vmax=np.log10(vmax_miri/3), cmap=cmap_miri)
            ax_jwst_f1130w.imshow(np.log10(cutout_jwst_f1130w.data), vmin=np.log10(vmax_miri/500),
                                  vmax=np.log10(vmax_miri/3), cmap=cmap_miri)
            ax_jwst_f2100w.imshow(np.log10(cutout_jwst_f2100w.data), vmin=np.log10(vmax_miri/500),
                                  vmax=np.log10(vmax_miri/3), cmap=cmap_miri)
            colorbar_label = r'log(S /[MJy / sr])'
        else:
            cb_hst = ax_hst_f275w.imshow(cutout_hst_f275w.data, vmin=-vmax_hst/10, vmax=vmax_hst, cmap=cmap_hst)
            ax_hst_f438w.imshow(cutout_hst_f438w.data, vmin=-vmax_hst/10, vmax=vmax_hst, cmap=cmap_hst)
            ax_hst_f555w.imshow(cutout_hst_f555w.data, vmin=-vmax_hst/10, vmax=vmax_hst, cmap=cmap_hst)
            ax_hst_f814w.imshow(cutout_hst_f814w.data, vmin=-vmax_hst/10, vmax=vmax_hst, cmap=cmap_hst)

            cb_nircam = ax_jwst_f200w.imshow(cutout_jwst_f200w.data, vmin=-vmax_nircam/10, vmax=vmax_nircam,
                                             cmap=cmap_nircam)
            ax_jwst_f300m.imshow(cutout_jwst_f300m.data, vmin=-vmax_nircam/10, vmax=vmax_nircam, cmap=cmap_nircam)
            ax_jwst_f335m.imshow(cutout_jwst_f335m.data, vmin=-vmax_nircam/10, vmax=vmax_nircam, cmap=cmap_nircam)
            ax_jwst_f360m.imshow(cutout_jwst_f360m.data, vmin=-vmax_nircam/10, vmax=vmax_nircam, cmap=cmap_nircam)

            cb_miri = ax_jwst_f770w.imshow(cutout_jwst_f770w.data, vmin=-vmax_miri/10, vmax=vmax_miri, cmap=cmap_miri)
            ax_jwst_f1000w.imshow(cutout_jwst_f1000w.data, vmin=-vmax_miri/10, vmax=vmax_miri, cmap=cmap_miri)
            ax_jwst_f1130w.imshow(cutout_jwst_f1130w.data, vmin=-vmax_miri/10, vmax=vmax_miri, cmap=cmap_miri)
            ax_jwst_f2100w.imshow(cutout_jwst_f2100w.data, vmin=-vmax_miri/10, vmax=vmax_miri, cmap=cmap_miri)
            colorbar_label = r'S [MJy / sr]'


        figure.colorbar(cb_hst, cax=ax_color_bar_hst, ticks=ticks_hst, orientation='vertical')
        ax_color_bar_hst.set_ylabel(colorbar_label, labelpad=2, fontsize=fontsize)
        ax_color_bar_hst.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False,
                                     labeltop=True, labelsize=fontsize)

        figure.colorbar(cb_nircam, cax=ax_color_bar_nircam, ticks=ticks_nircam, orientation='vertical')
        ax_color_bar_nircam.set_ylabel(colorbar_label, labelpad=2, fontsize=fontsize)
        ax_color_bar_nircam.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False,
                                        labeltop=True, labelsize=fontsize)

        figure.colorbar(cb_miri, cax=ax_color_bar_miri, ticks=ticks_miri, orientation='vertical')
        ax_color_bar_miri.set_ylabel(colorbar_label, labelpad=18, fontsize=fontsize)
        ax_color_bar_miri.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False,
                                      labeltop=True, labelsize=fontsize)

        VisualizeHelper.set_lim2cutout(ax=ax_hst_f275w, cutout=cutout_hst_f275w, cutout_pos=cutout_pos, edge_cut_ratio=100)
        VisualizeHelper.set_lim2cutout(ax=ax_hst_f438w, cutout=cutout_hst_f438w, cutout_pos=cutout_pos, edge_cut_ratio=100)
        VisualizeHelper.set_lim2cutout(ax=ax_hst_f555w, cutout=cutout_hst_f555w, cutout_pos=cutout_pos, edge_cut_ratio=100)
        VisualizeHelper.set_lim2cutout(ax=ax_hst_f814w, cutout=cutout_hst_f814w, cutout_pos=cutout_pos, edge_cut_ratio=100)

        VisualizeHelper.set_lim2cutout(ax=ax_jwst_f200w, cutout=cutout_jwst_f200w, cutout_pos=cutout_pos, edge_cut_ratio=100)
        VisualizeHelper.set_lim2cutout(ax=ax_jwst_f300m, cutout=cutout_jwst_f300m, cutout_pos=cutout_pos, edge_cut_ratio=100)
        VisualizeHelper.set_lim2cutout(ax=ax_jwst_f335m, cutout=cutout_jwst_f335m, cutout_pos=cutout_pos, edge_cut_ratio=100)
        VisualizeHelper.set_lim2cutout(ax=ax_jwst_f360m, cutout=cutout_jwst_f360m, cutout_pos=cutout_pos, edge_cut_ratio=100)

        VisualizeHelper.set_lim2cutout(ax=ax_jwst_f770w, cutout=cutout_jwst_f770w, cutout_pos=cutout_pos, edge_cut_ratio=100)
        VisualizeHelper.set_lim2cutout(ax=ax_jwst_f1000w, cutout=cutout_jwst_f1000w, cutout_pos=cutout_pos, edge_cut_ratio=100)
        VisualizeHelper.set_lim2cutout(ax=ax_jwst_f1130w, cutout=cutout_jwst_f1130w, cutout_pos=cutout_pos, edge_cut_ratio=100)
        VisualizeHelper.set_lim2cutout(ax=ax_jwst_f2100w, cutout=cutout_jwst_f2100w, cutout_pos=cutout_pos, edge_cut_ratio=100)

        if circ_pos:
            VisualizeHelper.plot_coord_circle(ax_hst_f275w, position=circ_pos, radius=circ_rad*u.arcsec, color='k')
            VisualizeHelper.plot_coord_circle(ax_hst_f438w, position=circ_pos, radius=circ_rad*u.arcsec, color='k')
            VisualizeHelper.plot_coord_circle(ax_hst_f555w, position=circ_pos, radius=circ_rad*u.arcsec, color='k')
            VisualizeHelper.plot_coord_circle(ax_hst_f814w, position=circ_pos, radius=circ_rad*u.arcsec, color='k')
            VisualizeHelper.plot_coord_circle(ax_jwst_f200w, position=circ_pos, radius=circ_rad*u.arcsec, color='k')
            VisualizeHelper.plot_coord_circle(ax_jwst_f300m, position=circ_pos, radius=circ_rad*u.arcsec, color='k')
            VisualizeHelper.plot_coord_circle(ax_jwst_f335m, position=circ_pos, radius=circ_rad*u.arcsec, color='k')
            VisualizeHelper.plot_coord_circle(ax_jwst_f360m, position=circ_pos, radius=circ_rad*u.arcsec, color='k')
            VisualizeHelper.plot_coord_circle(ax_jwst_f770w, position=circ_pos, radius=circ_rad*u.arcsec, color='k')
            VisualizeHelper.plot_coord_circle(ax_jwst_f1000w, position=circ_pos, radius=circ_rad*u.arcsec, color='k')
            VisualizeHelper.plot_coord_circle(ax_jwst_f1130w, position=circ_pos, radius=circ_rad*u.arcsec, color='k')
            VisualizeHelper.plot_coord_circle(ax_jwst_f2100w, position=circ_pos, radius=circ_rad*u.arcsec, color='k')


        text = cutout_hst_f275w.wcs.world_to_pixel(SkyCoord(cutout_pos.ra+name_ra_offset*u.arcsec, cutout_pos.dec+name_dec_offset*u.arcsec))
        ax_hst_f275w.text(text[0], text[1], hst_channel_list[0], color='k', fontsize=fontsize+2)

        text = cutout_hst_f438w.wcs.world_to_pixel(SkyCoord(cutout_pos.ra+name_ra_offset*u.arcsec, cutout_pos.dec+name_dec_offset*u.arcsec))
        ax_hst_f438w.text(text[0], text[1], hst_channel_list[1], color='k', fontsize=fontsize+2)

        text = cutout_hst_f555w.wcs.world_to_pixel(SkyCoord(cutout_pos.ra+name_ra_offset*u.arcsec, cutout_pos.dec+name_dec_offset*u.arcsec))
        ax_hst_f555w.text(text[0], text[1], hst_channel_list[2], color='k', fontsize=fontsize+2)

        text = cutout_hst_f814w.wcs.world_to_pixel(SkyCoord(cutout_pos.ra+name_ra_offset*u.arcsec, cutout_pos.dec+name_dec_offset*u.arcsec))
        ax_hst_f814w.text(text[0], text[1], hst_channel_list[3], color='k', fontsize=fontsize+2)


        text = cutout_jwst_f200w.wcs.world_to_pixel(SkyCoord(cutout_pos.ra+name_ra_offset*u.arcsec, cutout_pos.dec+name_dec_offset*u.arcsec))
        ax_jwst_f200w.text(text[0], text[1], nircam_channel_list[0], color='k', fontsize=fontsize+2)

        text = cutout_jwst_f300m.wcs.world_to_pixel(SkyCoord(cutout_pos.ra+name_ra_offset*u.arcsec, cutout_pos.dec+name_dec_offset*u.arcsec))
        ax_jwst_f300m.text(text[0], text[1], nircam_channel_list[1], color='k', fontsize=fontsize+2)

        text = cutout_jwst_f335m.wcs.world_to_pixel(SkyCoord(cutout_pos.ra+name_ra_offset*u.arcsec, cutout_pos.dec+name_dec_offset*u.arcsec))
        ax_jwst_f335m.text(text[0], text[1], nircam_channel_list[2], color='k', fontsize=fontsize+2)

        text = cutout_jwst_f360m.wcs.world_to_pixel(SkyCoord(cutout_pos.ra+name_ra_offset*u.arcsec, cutout_pos.dec+name_dec_offset*u.arcsec))
        ax_jwst_f360m.text(text[0], text[1], nircam_channel_list[3], color='k', fontsize=fontsize+2)


        text = cutout_jwst_f770w.wcs.world_to_pixel(SkyCoord(cutout_pos.ra+name_ra_offset*u.arcsec, cutout_pos.dec+name_dec_offset*u.arcsec))
        ax_jwst_f770w.text(text[0], text[1], miri_channel_list[0], color='k', fontsize=fontsize+2)

        text = cutout_jwst_f1000w.wcs.world_to_pixel(SkyCoord(cutout_pos.ra+name_ra_offset*u.arcsec, cutout_pos.dec+name_dec_offset*u.arcsec))
        ax_jwst_f1000w.text(text[0], text[1], miri_channel_list[1], color='k', fontsize=fontsize+2)

        text = cutout_jwst_f1130w.wcs.world_to_pixel(SkyCoord(cutout_pos.ra+name_ra_offset*u.arcsec, cutout_pos.dec+name_dec_offset*u.arcsec))
        ax_jwst_f1130w.text(text[0], text[1], miri_channel_list[2], color='k', fontsize=fontsize+2)

        text = cutout_jwst_f2100w.wcs.world_to_pixel(SkyCoord(cutout_pos.ra+name_ra_offset*u.arcsec, cutout_pos.dec+name_dec_offset*u.arcsec))
        ax_jwst_f2100w.text(text[0], text[1], miri_channel_list[3], color='k', fontsize=fontsize+2)


        ax_hst_f275w.tick_params(axis='both', which='both', width=3, length=7, direction='in', color='k', labelsize=fontsize-1)
        ax_hst_f438w.tick_params(axis='both', which='both', width=3, length=7, direction='in', color='k', labelsize=fontsize-1)
        ax_hst_f555w.tick_params(axis='both', which='both', width=3, length=7, direction='in', color='k', labelsize=fontsize-1)
        ax_hst_f814w.tick_params(axis='both', which='both', width=3, length=7, direction='in', color='k', labelsize=fontsize-1)
        ax_jwst_f200w.tick_params(axis='both', which='both', width=3, length=7, direction='in', color='k', labelsize=fontsize-1)
        ax_jwst_f300m.tick_params(axis='both', which='both', width=3, length=7, direction='in', color='k', labelsize=fontsize-1)
        ax_jwst_f335m.tick_params(axis='both', which='both', width=3, length=7, direction='in', color='k', labelsize=fontsize-1)
        ax_jwst_f360m.tick_params(axis='both', which='both', width=3, length=7, direction='in', color='k', labelsize=fontsize-1)
        ax_jwst_f770w.tick_params(axis='both', which='both', width=3, length=7, direction='in', color='k', labelsize=fontsize-1)
        ax_jwst_f1000w.tick_params(axis='both', which='both', width=3, length=7, direction='in', color='k', labelsize=fontsize-1)
        ax_jwst_f1130w.tick_params(axis='both', which='both', width=3, length=7, direction='in', color='k', labelsize=fontsize-1)
        ax_jwst_f2100w.tick_params(axis='both', which='both', width=3, length=7, direction='in', color='k', labelsize=fontsize-1)



        ax_hst_f275w.coords['dec'].set_ticklabel(rotation=90)
        ax_hst_f275w.coords['dec'].set_axislabel('DEC. (2000.0)', minpad=0.8, fontsize=fontsize)

        ax_jwst_f200w.coords['dec'].set_ticklabel(rotation=90)
        ax_jwst_f200w.coords['dec'].set_axislabel('DEC. (2000.0)', minpad=0.8, fontsize=fontsize)

        ax_jwst_f770w.coords['dec'].set_ticklabel(rotation=90)
        ax_jwst_f770w.coords['dec'].set_axislabel('DEC. (2000.0)', minpad=0.8, fontsize=fontsize)


        ax_jwst_f770w.coords['ra'].set_ticklabel(rotation=0)
        ax_jwst_f770w.coords['ra'].set_axislabel('R.A. (2000.0)', minpad=0.8, fontsize=fontsize)

        ax_jwst_f1000w.coords['ra'].set_ticklabel(rotation=0)
        ax_jwst_f1000w.coords['ra'].set_axislabel('R.A. (2000.0)', minpad=0.8, fontsize=fontsize)

        ax_jwst_f1130w.coords['ra'].set_ticklabel(rotation=0)
        ax_jwst_f1130w.coords['ra'].set_axislabel('R.A. (2000.0)', minpad=0.8, fontsize=fontsize)

        ax_jwst_f2100w.coords['ra'].set_ticklabel(rotation=0)
        ax_jwst_f2100w.coords['ra'].set_axislabel('R.A. (2000.0)', minpad=0.8, fontsize=fontsize)


        ax_hst_f275w.coords['ra'].set_ticklabel_visible(False)
        ax_hst_f275w.coords['ra'].set_axislabel(' ')

        ax_hst_f438w.coords['ra'].set_ticklabel_visible(False)
        ax_hst_f438w.coords['ra'].set_axislabel(' ')

        ax_hst_f555w.coords['ra'].set_ticklabel_visible(False)
        ax_hst_f555w.coords['ra'].set_axislabel(' ')

        ax_hst_f814w.coords['ra'].set_ticklabel_visible(False)
        ax_hst_f814w.coords['ra'].set_axislabel(' ')


        ax_jwst_f200w.coords['ra'].set_ticklabel_visible(False)
        ax_jwst_f200w.coords['ra'].set_axislabel(' ')

        ax_jwst_f300m.coords['ra'].set_ticklabel_visible(False)
        ax_jwst_f300m.coords['ra'].set_axislabel(' ')

        ax_jwst_f335m.coords['ra'].set_ticklabel_visible(False)
        ax_jwst_f335m.coords['ra'].set_axislabel(' ')

        ax_jwst_f360m.coords['ra'].set_ticklabel_visible(False)
        ax_jwst_f360m.coords['ra'].set_axislabel(' ')


        ax_hst_f438w.coords['dec'].set_ticklabel_visible(False)
        ax_hst_f438w.coords['dec'].set_axislabel(' ')

        ax_hst_f555w.coords['dec'].set_ticklabel_visible(False)
        ax_hst_f555w.coords['dec'].set_axislabel(' ')

        ax_hst_f814w.coords['dec'].set_ticklabel_visible(False)
        ax_hst_f814w.coords['dec'].set_axislabel(' ')

        ax_jwst_f300m.coords['dec'].set_ticklabel_visible(False)
        ax_jwst_f300m.coords['dec'].set_axislabel(' ')

        ax_jwst_f335m.coords['dec'].set_ticklabel_visible(False)
        ax_jwst_f335m.coords['dec'].set_axislabel(' ')

        ax_jwst_f360m.coords['dec'].set_ticklabel_visible(False)
        ax_jwst_f360m.coords['dec'].set_axislabel(' ')

        ax_jwst_f1000w.coords['dec'].set_ticklabel_visible(False)
        ax_jwst_f1000w.coords['dec'].set_axislabel(' ')

        ax_jwst_f1130w.coords['dec'].set_ticklabel_visible(False)
        ax_jwst_f1130w.coords['dec'].set_axislabel(' ')

        ax_jwst_f2100w.coords['dec'].set_ticklabel_visible(False)
        ax_jwst_f2100w.coords['dec'].set_axislabel(' ')



        ax_hst_f275w.coords['ra'].set_ticks(number=ra_tick_num)
        ax_hst_f438w.coords['ra'].set_ticks(number=ra_tick_num)
        ax_hst_f555w.coords['ra'].set_ticks(number=ra_tick_num)
        ax_hst_f814w.coords['ra'].set_ticks(number=ra_tick_num)
        ax_jwst_f200w.coords['ra'].set_ticks(number=ra_tick_num)
        ax_jwst_f300m.coords['ra'].set_ticks(number=ra_tick_num)
        ax_jwst_f335m.coords['ra'].set_ticks(number=ra_tick_num)
        ax_jwst_f360m.coords['ra'].set_ticks(number=ra_tick_num)
        ax_jwst_f770w.coords['ra'].set_ticks(number=ra_tick_num)
        ax_jwst_f1000w.coords['ra'].set_ticks(number=ra_tick_num)
        ax_jwst_f1130w.coords['ra'].set_ticks(number=ra_tick_num)
        ax_jwst_f2100w.coords['ra'].set_ticks(number=ra_tick_num)

        ax_hst_f275w.coords['ra'].display_minor_ticks(True)
        ax_hst_f438w.coords['ra'].display_minor_ticks(True)
        ax_hst_f555w.coords['ra'].display_minor_ticks(True)
        ax_hst_f814w.coords['ra'].display_minor_ticks(True)
        ax_jwst_f200w.coords['ra'].display_minor_ticks(True)
        ax_jwst_f300m.coords['ra'].display_minor_ticks(True)
        ax_jwst_f335m.coords['ra'].display_minor_ticks(True)
        ax_jwst_f360m.coords['ra'].display_minor_ticks(True)
        ax_jwst_f770w.coords['ra'].display_minor_ticks(True)
        ax_jwst_f1000w.coords['ra'].display_minor_ticks(True)
        ax_jwst_f1130w.coords['ra'].display_minor_ticks(True)
        ax_jwst_f2100w.coords['ra'].display_minor_ticks(True)


        ax_hst_f275w.coords['dec'].set_ticks(number=dec_tick_num)
        ax_hst_f438w.coords['dec'].set_ticks(number=dec_tick_num)
        ax_hst_f555w.coords['dec'].set_ticks(number=dec_tick_num)
        ax_hst_f814w.coords['dec'].set_ticks(number=dec_tick_num)
        ax_jwst_f200w.coords['dec'].set_ticks(number=dec_tick_num)
        ax_jwst_f300m.coords['dec'].set_ticks(number=dec_tick_num)
        ax_jwst_f335m.coords['dec'].set_ticks(number=dec_tick_num)
        ax_jwst_f360m.coords['dec'].set_ticks(number=dec_tick_num)
        ax_jwst_f770w.coords['dec'].set_ticks(number=dec_tick_num)
        ax_jwst_f1000w.coords['dec'].set_ticks(number=dec_tick_num)
        ax_jwst_f1130w.coords['dec'].set_ticks(number=dec_tick_num)
        ax_jwst_f2100w.coords['dec'].set_ticks(number=dec_tick_num)


        ax_hst_f275w.coords['dec'].display_minor_ticks(True)
        ax_hst_f438w.coords['dec'].display_minor_ticks(True)
        ax_hst_f555w.coords['dec'].display_minor_ticks(True)
        ax_hst_f814w.coords['dec'].display_minor_ticks(True)
        ax_jwst_f200w.coords['dec'].display_minor_ticks(True)
        ax_jwst_f300m.coords['dec'].display_minor_ticks(True)
        ax_jwst_f335m.coords['dec'].display_minor_ticks(True)
        ax_jwst_f360m.coords['dec'].display_minor_ticks(True)
        ax_jwst_f770w.coords['dec'].display_minor_ticks(True)
        ax_jwst_f1000w.coords['dec'].display_minor_ticks(True)
        ax_jwst_f1130w.coords['dec'].display_minor_ticks(True)
        ax_jwst_f2100w.coords['dec'].display_minor_ticks(True)

        return figure

    @staticmethod
    def draw_box(ax, wcs, coord, box_size, color='k', linewidth=2):

        if isinstance(box_size, tuple):
            box_size = box_size * u.arcsec
        elif isinstance(box_size, float) | isinstance(box_size, int):
            box_size = (box_size, box_size) * u.arcsec
        else:
            raise KeyError('cutout_size must be float or tuple')

        top_left_pix = wcs.world_to_pixel(SkyCoord(ra=coord.ra + box_size[0] / 2, dec=coord.dec + box_size[1] / 2))
        top_right_pix = wcs.world_to_pixel(SkyCoord(ra=coord.ra + box_size[0] / 2, dec=coord.dec - box_size[1] / 2))
        bottom_left_pix = wcs.world_to_pixel(SkyCoord(ra=coord.ra - box_size[0] / 2, dec=coord.dec + box_size[1] / 2))
        bottom_right_pix = wcs.world_to_pixel(SkyCoord(ra=coord.ra - box_size[0] / 2, dec=coord.dec - box_size[1] / 2))

        ax.plot([top_left_pix[0], top_right_pix[0]], [top_left_pix[1], top_right_pix[1]], color=color, linewidth=linewidth)
        ax.plot([bottom_left_pix[0], bottom_right_pix[0]], [bottom_left_pix[1], bottom_right_pix[1]], color=color, linewidth=linewidth)
        ax.plot([top_left_pix[0], bottom_left_pix[0]], [top_left_pix[1], bottom_left_pix[1]], color=color, linewidth=linewidth)
        ax.plot([top_right_pix[0], bottom_right_pix[0]], [top_right_pix[1], bottom_right_pix[1]], color=color, linewidth=linewidth)

    @staticmethod
    def plot_coord_circle(ax, position, radius, color='g', linestyle='-', linewidth=3, alpha=1.):
        circle = SphericalCircle(position, radius,
                                 edgecolor=color, facecolor='none', linewidth=linewidth, linestyle=linestyle,
                                 alpha=alpha, transform=ax.get_transform('icrs'))

        ax.add_patch(circle)

    @staticmethod
    def erase_axis(ax):
        ax.coords['ra'].set_ticklabel_visible(False)
        ax.coords['ra'].set_axislabel(' ')
        ax.coords['dec'].set_ticklabel_visible(False)
        ax.coords['dec'].set_axislabel(' ')
        ax.tick_params(axis='both', which='both', width=0.00001, direction='in', color='k')

    @staticmethod
    def plot_cigale_sed_results(cigale_flux_file_name, hdu_best_model_file_name, flux_fitted_model_file_name,
                                hst_file_name_list, nircam_file_name_list, miri_file_name_list,
                                cutout_pos, cutout_size,
                                hdu_number_hst=0, hdu_number_nircam='SCI', hdu_number_miri=0,
                                filter_list=None, cigale_flux_file_col_name_ord=None, index_cigale_table=0,
                                cigale_filt_names=None, cigale_logo_file_name=None, circle_rad=0.2,
                                filter_colors=None):

        if filter_list is None:
            filter_list = np.array(['F275W', 'F336W', 'F438W', 'F555W', 'F814W',
                                    'F200W', 'F300M', 'F335M', 'F360M',
                                    'F770W', 'F1000W', 'F1130W', 'F2100W'])
        if cigale_flux_file_col_name_ord is None:
            cigale_flux_file_col_name_ord = np.array(['id', 'redshift', 'distance',
                                                      'F275W', 'F275W_err', 'F336W', 'F336W_err', 'F438W', 'F438W_err',
                                                      'F555W', 'F555W_err', 'F814W', 'F814W_err', 'F200W', 'F200W_err',
                                                      'F300M', 'F300M_err', 'F335M', 'F335M_err',  'F360M', 'F360M_err',
                                                      'F770W', 'F770W_err', 'F1000W', 'F1000W_err',
                                                      'F1130W', 'F1130W_err', 'F2100W', 'F2100W_err'])
        if cigale_filt_names is None:
            cigale_filt_names = np.array(['F275W_UVIS_CHIP2', 'F336W_UVIS_CHIP2', 'F438W_UVIS_CHIP2',
                                          'F555W_UVIS_CHIP2', 'F814W_UVIS_CHIP2', 'jwst.nircam.F200W',
                                          'jwst.nircam.F300M', 'jwst.nircam.F335M', 'jwst.nircam.F360M',
                                          'jwst.miri.F770W', 'jwst.miri.F1000W', 'jwst.miri.F1130W',
                                          'jwst.miri.F2100W'])

        if filter_colors is None:
            filter_colors = np.array(['k', 'k', 'k', 'k', 'k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                                      'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray'])

        cutout_hst_f275w = VisualizeHelper.get_hst_cutout_from_file(file_name=hst_file_name_list[0],
                                                                    hdu_number=hdu_number_hst, cutout_pos=cutout_pos,
                                                                    cutout_size=cutout_size, rescaling='MJy/sr')
        cutout_hst_f336w = VisualizeHelper.get_hst_cutout_from_file(file_name=hst_file_name_list[1],
                                                                    hdu_number=hdu_number_hst, cutout_pos=cutout_pos,
                                                                    cutout_size=cutout_size, rescaling='MJy/sr')
        cutout_hst_f438w = VisualizeHelper.get_hst_cutout_from_file(file_name=hst_file_name_list[2],
                                                                    hdu_number=hdu_number_hst, cutout_pos=cutout_pos,
                                                                    cutout_size=cutout_size, rescaling='MJy/sr')
        cutout_hst_f555w = VisualizeHelper.get_hst_cutout_from_file(file_name=hst_file_name_list[3],
                                                                    hdu_number=hdu_number_hst, cutout_pos=cutout_pos,
                                                                    cutout_size=cutout_size, rescaling='MJy/sr')
        cutout_hst_f814w = VisualizeHelper.get_hst_cutout_from_file(file_name=hst_file_name_list[4],
                                                                    hdu_number=hdu_number_hst, cutout_pos=cutout_pos,
                                                                    cutout_size=cutout_size, rescaling='MJy/sr')

        cutout_jwst_f200w = VisualizeHelper.get_jwst_cutout_from_file(file_name=nircam_file_name_list[0],
                                                                      hdu_number=hdu_number_nircam,
                                                                      cutout_pos=cutout_pos, cutout_size=cutout_size,
                                                                      rescaling='MJy/sr')
        cutout_jwst_f300m = VisualizeHelper.get_jwst_cutout_from_file(file_name=nircam_file_name_list[1],
                                                                      hdu_number=hdu_number_nircam,
                                                                      cutout_pos=cutout_pos, cutout_size=cutout_size,
                                                                      rescaling='MJy/sr')
        cutout_jwst_f335m = VisualizeHelper.get_jwst_cutout_from_file(file_name=nircam_file_name_list[2],
                                                                      hdu_number=hdu_number_nircam,
                                                                      cutout_pos=cutout_pos, cutout_size=cutout_size,
                                                                      rescaling='MJy/sr')
        cutout_jwst_f360m = VisualizeHelper.get_jwst_cutout_from_file(file_name=nircam_file_name_list[3],
                                                                      hdu_number=hdu_number_nircam,
                                                                      cutout_pos=cutout_pos, cutout_size=cutout_size,
                                                                      rescaling='MJy/sr')

        cutout_jwst_f770w = VisualizeHelper.get_jwst_cutout_from_file(file_name=miri_file_name_list[0],
                                                                      hdu_number=hdu_number_miri, cutout_pos=cutout_pos,
                                                                      cutout_size=cutout_size, rescaling='MJy/sr')
        cutout_jwst_f1000w = VisualizeHelper.get_jwst_cutout_from_file(file_name=miri_file_name_list[1],
                                                                       hdu_number=hdu_number_miri,
                                                                       cutout_pos=cutout_pos, cutout_size=cutout_size,
                                                                       rescaling='MJy/sr')
        cutout_jwst_f1130w = VisualizeHelper.get_jwst_cutout_from_file(file_name=miri_file_name_list[2],
                                                                       hdu_number=hdu_number_miri,
                                                                       cutout_pos=cutout_pos, cutout_size=cutout_size,
                                                                       rescaling='MJy/sr')
        cutout_jwst_f2100w = VisualizeHelper.get_jwst_cutout_from_file(file_name=miri_file_name_list[3],
                                                                       hdu_number=hdu_number_miri,
                                                                       cutout_pos=cutout_pos, cutout_size=cutout_size,
                                                                       rescaling='MJy/sr')

        # get the observed fluxed which entered into the CIGAL fit
        cigale_flux_file = np.genfromtxt(cigale_flux_file_name)
        # get all the fluxes from file
        flux_list = []
        flux_err_list = []
        if cigale_flux_file.ndim == 1:
            for filter_name in filter_list:
                flux_list.append(cigale_flux_file[cigale_flux_file_col_name_ord == filter_name])
                flux_err_list.append(cigale_flux_file[cigale_flux_file_col_name_ord == filter_name + '_err'])
            flux_list = np.array(flux_list).T[0]
            flux_err_list = np.array(flux_err_list).T[0]
        elif cigale_flux_file.ndim == 2:
            for filter_name in filter_list:
                flux_list.append(cigale_flux_file[:, cigale_flux_file_col_name_ord ==
                                                     filter_name].T[0][index_cigale_table])
                flux_err_list.append(cigale_flux_file[:, cigale_flux_file_col_name_ord ==
                                                         filter_name + '_err'].T[0][index_cigale_table])
            flux_list = np.array(flux_list).T
            flux_err_list = np.array(flux_err_list).T
        else:
            raise IndexError('There is something wrong with the dimension of the Cigale flux array')

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

        wave_list = np.array([wave_f275w, wave_f336w, wave_f438w, wave_f555w, wave_f814w, wave_f200w, wave_f300m,
                              wave_f335m, wave_f360m, wave_f770w, wave_f1000w, wave_f1130w, wave_f2100w])

        hdu_best_model = fits.open(hdu_best_model_file_name)
        data = hdu_best_model[1].data
        header = hdu_best_model[1].header
        for names in hdu_best_model[1].header.keys():
            print(names, hdu_best_model[1].header[names])

        results = fits.open(flux_fitted_model_file_name)
        # print(results.info())
        results_data = results[1].data

        # for names in results[1].header.keys():
        #     print(names, results[1].header[names])

        wavelength_spec = data["wavelength"] * 1e-3
        surf = 4.0 * np.pi * float(header["universe.luminosity_distance"]) ** 2
        fact = 1e29 * 1e-3 * wavelength_spec ** 2 / speed_of_light / surf

        spectrum = data['Fnu']
        wavelength = data['wavelength']

        stellar_spectrum = (data["stellar.young"] + data["stellar.old"] +
                            data["attenuation.stellar.young"] + data["attenuation.stellar.old"]) * fact
        stellar_spectrum_unattenuated = (data["stellar.young"] + data["stellar.old"]) * fact
        dust_spectrum = (data["dust"]) * fact


        # plotting
        fig = plt.figure(figsize=(30, 12))
        fontsize = 33

        ax_fit = fig.add_axes([0.06, 0.3, 0.935, 0.65])
        best_fit_residuals = fig.add_axes([0.06, 0.1, 0.935, 0.2])

        ax_image_f275w = fig.add_axes([0.03, 0.77, 0.13, 0.13], projection=cutout_hst_f275w.wcs)
        ax_image_f336w = fig.add_axes([0.09, 0.77, 0.13, 0.13], projection=cutout_hst_f336w.wcs)
        ax_image_f438w = fig.add_axes([0.15, 0.77, 0.13, 0.13], projection=cutout_hst_f438w.wcs)
        ax_image_f555w = fig.add_axes([0.21, 0.77, 0.13, 0.13], projection=cutout_hst_f555w.wcs)
        ax_image_f814w = fig.add_axes([0.27, 0.77, 0.13, 0.13], projection=cutout_hst_f814w.wcs)

        ax_image_f200w = fig.add_axes([0.36, 0.77, 0.13, 0.13], projection=cutout_jwst_f200w.wcs)
        ax_image_f300m = fig.add_axes([0.44, 0.77, 0.13, 0.13], projection=cutout_jwst_f300m.wcs)
        ax_image_f335m = fig.add_axes([0.5, 0.77, 0.13, 0.13], projection=cutout_jwst_f335m.wcs)
        ax_image_f360m = fig.add_axes([0.56, 0.77, 0.13, 0.13], projection=cutout_jwst_f360m.wcs)

        ax_image_f770w = fig.add_axes([0.65, 0.77, 0.13, 0.13], projection=cutout_jwst_f770w.wcs)
        ax_image_f1000w = fig.add_axes([0.71, 0.77, 0.13, 0.13], projection=cutout_jwst_f1000w.wcs)
        ax_image_f1130w = fig.add_axes([0.77, 0.77, 0.13, 0.13], projection=cutout_jwst_f1130w.wcs)
        ax_image_f2100w = fig.add_axes([0.86, 0.77, 0.13, 0.13], projection=cutout_jwst_f2100w.wcs)

        if cigale_logo_file_name is not None:
            ax_cigale_logo = fig.add_axes([0.885, 0.31, 0.15, 0.15])
            ax_cigale_logo.imshow(mpimg.imread(cigale_logo_file_name))
            ax_cigale_logo.axis('off')

        ax_fit.plot(wavelength * 1e-3, stellar_spectrum, linewidth=3, color='yellow', label='Stellar attenuated')
        ax_fit.plot(wavelength * 1e-3, stellar_spectrum_unattenuated, linewidth=3, linestyle='--', color='b',
                    label='Stellar unattenuated')
        ax_fit.plot(wavelength * 1e-3, dust_spectrum, linewidth=3, color='r', label='Dust emission')
        ax_fit.plot(wavelength * 1e-3, spectrum, color='k', linewidth=3)

        for filter_idx in filter_list:
            if flux_err_list[filter_list == filter_idx] == -1:
                ax_fit.errorbar(wave_list[filter_list == filter_idx] * 1e-3, flux_list[filter_list == filter_idx],
                                xerr=wave_list[filter_list == filter_idx] * 2.5e-5,
                                ecolor=filter_colors[filter_list == filter_idx][0], elinewidth=5, capsize=1)
                ax_fit.errorbar(wave_list[filter_list == filter_idx] * 1e-3, flux_list[filter_list == filter_idx],
                                yerr=flux_list[filter_list == filter_idx] * 6e-1,
                                ecolor=filter_colors[filter_list == filter_idx][0],
                                elinewidth=5, capsize=10, uplims=True, xlolims=False)
            else:
                ax_fit.errorbar(wave_list[filter_list == filter_idx] * 1e-3, flux_list[filter_list == filter_idx],
                                yerr=flux_err_list[filter_list == filter_idx], ms=15, ecolor='k', fmt='o',
                                color=filter_colors[filter_list == filter_idx][0])
                flux_best_model = results_data['bayes.' + cigale_filt_names[filter_list == filter_idx][0]][0]
                best_fit_residuals.errorbar(wave_list[filter_list == filter_idx] * 1e-3,
                                            (flux_list[filter_list == filter_idx] - flux_best_model)
                                            / flux_list[filter_list == filter_idx],
                                            yerr=(flux_err_list[filter_list == filter_idx] * flux_best_model /
                                                  (flux_list[filter_list == filter_idx] ** 2)),
                                            fmt='.', ms=20, color=filter_colors[filter_list == filter_idx][0])

        # get values from fit to plot:
        reduced_chi2 = results[1].data['best.reduced_chi_square']
        age_star = hdu_best_model[1].header['sfh.age']
        # e_bv = hdu_best_model[1].header['attenuation.E_BV']
        att_a550 = hdu_best_model[1].header['attenuation.A550']

        met_star = hdu_best_model[1].header['stellar.metallicity']
        stellar_mass = hdu_best_model[1].header['stellar.m_star']
        print('reduced_chi2 ', reduced_chi2)
        print('age_star ', age_star)
        print('att_a550 ', att_a550)
        print('met_star ', met_star)
        print('stellar_mass ', stellar_mass)

        ax_fit.set_ylabel(r'S$_{\nu}$ (mJy)', fontsize=fontsize)
        best_fit_residuals.set_xlabel(r'Observed ${\lambda}$ ($\mu$m)', labelpad=-2, fontsize=fontsize)
        best_fit_residuals.set_ylabel('  Relative \n residuals', labelpad=-9, fontsize=fontsize - 2)
        best_fit_residuals.text(5e3, 0.4, '(Obs - Mod) / Obs', color='k', fontsize=fontsize - 4)
        # legend
        ax_fit.legend(frameon=False, fontsize=fontsize-6, bbox_to_anchor=[0.01, 0.45])
        # fit parameters
        ax_fit.text(1.1, 500, r'$\chi^{2}$/Ndof = %.1f' % reduced_chi2, fontsize=fontsize-6)
        ax_fit.text(1.1, 50, 'stellar age = %i Myr' % int(age_star), fontsize=fontsize-6)
        ax_fit.text(1.1, 5, r'M$_{*}$ = %.1f $\times$ 10$^{6}$ M$_{\odot}$' % (float(stellar_mass)*1e-6), fontsize=fontsize-6)
        ax_fit.text(1.1, 0.5, 'stellar metallicity = %.2f' % float(met_star), fontsize=fontsize-6)
        ax_fit.text(1.1, 0.05, r'A$_{550}$ = %.1f' % float(att_a550), fontsize=fontsize-6)


        best_fit_residuals.text(13.5e0, -0.5, '(Obs - Mod) / Obs', color='k', fontsize=fontsize - 4)
        ax_fit.tick_params(axis='both', which='both', width=4, length=5, direction='in', pad=10, labelsize=fontsize)
        best_fit_residuals.tick_params(axis='both', which='both', width=4, length=5, top=True, direction='in', labelsize=fontsize-4)
        best_fit_residuals.tick_params(axis='x', which='major', pad=10)
        ax_fit.set_xlim(200 * 1e-3, 3e1)
        best_fit_residuals.set_xlim(200 * 1e-3, 3e1)
        ax_fit.set_ylim(0.0000009, 5e7)
        best_fit_residuals.set_ylim(-0.8, 0.8)
        ax_fit.set_xscale('log')
        ax_fit.set_yscale('log')
        best_fit_residuals.plot([90 * 1e-3, 5e6], [0, 0], linestyle='--', linewidth=2, color='k')
        best_fit_residuals.set_xscale('log')

        ax_image_f275w.imshow(cutout_hst_f275w.data, cmap='Greys', origin='lower')
        ax_image_f336w.imshow(cutout_hst_f336w.data, cmap='Greys', origin='lower')
        ax_image_f438w.imshow(cutout_hst_f438w.data, cmap='Greys', origin='lower')
        ax_image_f555w.imshow(cutout_hst_f555w.data, cmap='Greys', origin='lower')
        ax_image_f814w.imshow(cutout_hst_f814w.data, cmap='Greys', origin='lower')

        ax_image_f200w.imshow(cutout_jwst_f200w.data, cmap='Greys', origin='lower')
        ax_image_f300m.imshow(cutout_jwst_f300m.data, cmap='Greys', origin='lower')
        ax_image_f335m.imshow(cutout_jwst_f335m.data, cmap='Greys', origin='lower')
        ax_image_f360m.imshow(cutout_jwst_f360m.data, cmap='Greys', origin='lower')

        ax_image_f770w.imshow(cutout_jwst_f770w.data, cmap='Greys', origin='lower')
        ax_image_f1000w.imshow(cutout_jwst_f1000w.data, cmap='Greys', origin='lower')
        ax_image_f1130w.imshow(cutout_jwst_f1130w.data, cmap='Greys', origin='lower')
        ax_image_f2100w.imshow(cutout_jwst_f2100w.data, cmap='Greys', origin='lower')

        VisualizeHelper.plot_coord_circle(ax=ax_image_f275w, position=cutout_pos, radius=circle_rad * u.arcsec,
                                          color='r', linewidth=2)
        VisualizeHelper.plot_coord_circle(ax=ax_image_f336w, position=cutout_pos, radius=circle_rad * u.arcsec,
                                          color='r', linewidth=2)
        VisualizeHelper.plot_coord_circle(ax=ax_image_f438w, position=cutout_pos, radius=circle_rad * u.arcsec,
                                          color='r', linewidth=2)
        VisualizeHelper.plot_coord_circle(ax=ax_image_f555w, position=cutout_pos, radius=circle_rad * u.arcsec,
                                          color='r', linewidth=2)
        VisualizeHelper.plot_coord_circle(ax=ax_image_f814w, position=cutout_pos, radius=circle_rad * u.arcsec,
                                          color='r', linewidth=2)

        VisualizeHelper.plot_coord_circle(ax=ax_image_f200w, position=cutout_pos, radius=circle_rad * u.arcsec,
                                          color='r', linewidth=2)
        VisualizeHelper.plot_coord_circle(ax=ax_image_f300m, position=cutout_pos, radius=circle_rad * u.arcsec,
                                          color='r', linewidth=2)
        VisualizeHelper.plot_coord_circle(ax=ax_image_f335m, position=cutout_pos, radius=circle_rad * u.arcsec,
                                          color='r', linewidth=2)
        VisualizeHelper.plot_coord_circle(ax=ax_image_f360m, position=cutout_pos, radius=circle_rad * u.arcsec,
                                          color='r', linewidth=2)

        VisualizeHelper.plot_coord_circle(ax=ax_image_f770w, position=cutout_pos, radius=circle_rad * u.arcsec,
                                          color='r', linewidth=2)
        VisualizeHelper.plot_coord_circle(ax=ax_image_f1000w, position=cutout_pos, radius=circle_rad * u.arcsec,
                                          color='r', linewidth=2)
        VisualizeHelper.plot_coord_circle(ax=ax_image_f1130w, position=cutout_pos, radius=circle_rad * u.arcsec,
                                          color='r', linewidth=2)
        VisualizeHelper.plot_coord_circle(ax=ax_image_f2100w, position=cutout_pos, radius=circle_rad * u.arcsec,
                                          color='r', linewidth=2)

        ax_image_f275w.set_title('F275W', fontsize=fontsize-5, color='k')
        ax_image_f336w.set_title('F336W', fontsize=fontsize-5, color='k')
        ax_image_f438w.set_title('F438W', fontsize=fontsize-5, color='k')
        ax_image_f555w.set_title('F555W', fontsize=fontsize-5, color='k')
        ax_image_f814w.set_title('F814W', fontsize=fontsize-5, color='k')
        ax_image_f200w.set_title('F200W', fontsize=fontsize-5, color='tab:blue')
        ax_image_f300m.set_title('F300M', fontsize=fontsize-5, color='tab:orange')
        ax_image_f335m.set_title('F335M', fontsize=fontsize-5, color='tab:green')
        ax_image_f360m.set_title('F360M', fontsize=fontsize-5, color='tab:red')
        ax_image_f770w.set_title('F770W', fontsize=fontsize-5, color='tab:purple')
        ax_image_f1000w.set_title('F1000W', fontsize=fontsize-5, color='tab:brown')
        ax_image_f1130w.set_title('F1130W', fontsize=fontsize-5, color='tab:pink')
        ax_image_f2100w.set_title('F2100W', fontsize=fontsize-5, color='tab:gray')

        VisualizeHelper.erase_axis(ax_image_f275w)
        VisualizeHelper.erase_axis(ax_image_f336w)
        VisualizeHelper.erase_axis(ax_image_f438w)
        VisualizeHelper.erase_axis(ax_image_f555w)
        VisualizeHelper.erase_axis(ax_image_f814w)
        VisualizeHelper.erase_axis(ax_image_f200w)
        VisualizeHelper.erase_axis(ax_image_f300m)
        VisualizeHelper.erase_axis(ax_image_f335m)
        VisualizeHelper.erase_axis(ax_image_f360m)
        VisualizeHelper.erase_axis(ax_image_f770w)
        VisualizeHelper.erase_axis(ax_image_f1000w)
        VisualizeHelper.erase_axis(ax_image_f1130w)
        VisualizeHelper.erase_axis(ax_image_f2100w)

        return fig

    @staticmethod
    def rebin_2d_array(a, shape):
        sh = shape[0], a.shape[0]//shape[0], shape[1], a.shape[1]//shape[1]
        return a.reshape(sh).mean(-1).mean(1)

    @staticmethod
    def get_photutil_catalog(img, fwhm=2.0, thresh=3.0, cut_lim=0.05):

        mean, median, std = sigma_clipped_stats(img, sigma=3.0)
        daofind = DAOStarFinder(fwhm=fwhm, threshold=thresh*std)
        sources = daofind(img - median)
        if sources is None:
            daofind = DAOStarFinder(fwhm=fwhm*10, threshold=thresh*std)
            sources = daofind(img - median)
        if sources is None:
            daofind = DAOStarFinder(fwhm=fwhm*20, threshold=thresh*std)
            sources = daofind(img - median)
        if sources is None:
            daofind = DAOStarFinder(fwhm=fwhm*30, threshold=thresh*std)
            sources = daofind(img - median)

        if sources is None:
            return None

        for col in sources.colnames:
            sources[col].info.format = '%.8g'  # for consistent table output

        x_coords, y_coords = sources['xcentroid'], sources['ycentroid']
        # plt.imshow(img)
        # plt.scatter(x_coords, y_coords, color='r')
        # plt.show()
        # exit()
        shape = img.shape
        # exclude sources at the limit of the image
        sources_at_limit = ((x_coords < cut_lim*shape[0]) | (x_coords > (1-cut_lim)*shape[0]) |
                            (y_coords < cut_lim*shape[1]) | (y_coords > (1-cut_lim)*shape[1]))

        return sources[~sources_at_limit]

    @staticmethod
    def get_brightest_pixel_coordinates(img):
        max_index = np.where(img == img.max())
        return np.array([max_index[0], max_index[1]])

    @staticmethod
    def create_scarlet_obs(img, wcs, psf, channel, weights=None):
        # reshape the image shape
        n1, n2 = np.shape(img)
        img = img.reshape(1, n1, n2).byteswap().newbyteorder()
        # create observation object
        obs = scarlet.Observation(data=img, wcs=wcs, psf=psf, channels=channel, weights=weights)

        model_psf = scarlet.GaussianPSF(sigma=(0.9,))
        model_frame = scarlet.Frame(shape=img.shape, wcs=wcs, psf=model_psf, channels=channel)
        obs.match(model_frame)

        return obs, model_frame

    @staticmethod
    def init_sources(obs, model_frame, ra_dec_point_src=None, ra_dec_extend_src=None, extended_box_size=60):

        if (ra_dec_point_src is None) & (ra_dec_extend_src is None):
            raise KeyError('Either point_src or extend_src must be not None')

        src = []
        if ra_dec_point_src is not None:
            if len(ra_dec_point_src) != 0:
                if ra_dec_point_src.ndim == 1:
                    new_src = scarlet.PointSource(model_frame, ra_dec_point_src, obs)
                    src.append(new_src)
                else:
                    for ra_dec in ra_dec_point_src:
                        new_src = scarlet.PointSource(model_frame, ra_dec, obs)
                        src.append(new_src)
        if ra_dec_extend_src is not None:
            if len(ra_dec_extend_src) != 0:
                if ra_dec_extend_src.ndim == 1:
                    new_src = scarlet.CompactExtendedSource(model_frame, ra_dec_extend_src, obs, boxsize=extended_box_size)
                    src.append(new_src)
                else:
                    for ra_dec in ra_dec_extend_src:
                        new_src = scarlet.CompactExtendedSource(model_frame, ra_dec, obs, boxsize=extended_box_size)
                        src.append(new_src)
        scarlet.initialization.set_spectra_to_match(src, obs)

        return src

    @staticmethod
    def get_scarlet_model(obs, model_frame, scarlet_sources):
        model = np.zeros(model_frame.shape)
        for src in scarlet_sources:
            model += src.get_model(frame=model_frame)
        return obs.render(model)

    @staticmethod
    def get_scarlet_src_flux(band, fit_result_dict, mask):
        flux = 0
        for src in np.array(fit_result_dict['scarlet_sources_%s' % band])[mask]:
            flux += scarlet.measure.flux(component=src)
        return flux

    @staticmethod
    def get_scarlet_residuals(obs, model_frame, scarlet_sources):
        model = VisualizeHelper.get_scarlet_model(obs=obs, model_frame=model_frame, scarlet_sources=scarlet_sources)
        return obs.data - model

    @staticmethod
    def scarlet_fit_sources2img(obs, model_frame, ra_dec_point_src, ra_dec_extended_src,
                                box_size_extended=100, fitting_steps=50, fitting_rel_err=1e-14):

        scarlet_sources = VisualizeHelper.init_sources(obs=obs, model_frame=model_frame,
                                                       ra_dec_point_src=ra_dec_point_src,
                                                       ra_dec_extend_src=ra_dec_extended_src,
                                                       extended_box_size=box_size_extended)
        # scarlet.initialization.set_spectra_to_match(scarlet_sources, obs)
        blend = scarlet.Blend(scarlet_sources, obs)
        blend.fit(fitting_steps, e_rel=fitting_rel_err)
        # it, logL = blend.fit(fitting_steps, e_rel=fitting_rel_err)
        # print(f"scarlet ran for {it} iterations to logL = {logL}")
        # scarlet.display.show_likelihood(blend)
        # plt.show()

        return obs, model_frame, scarlet_sources

    @staticmethod
    def find_point_src(img, wcs, psf, channel, fwhm_point=2.0, thresh_point=3.0):

        src_catalog_point = VisualizeHelper.get_photutil_catalog(img, fwhm=fwhm_point, thresh=thresh_point)
        obs, model_frame = VisualizeHelper.create_scarlet_obs(img=img, wcs=wcs, psf=psf, channel=channel)
        pixel_point_src = np.stack((src_catalog_point['ycentroid'], src_catalog_point['xcentroid']), axis=1)
        ra_dec_point_src = obs.get_sky_coord(pixel_point_src)

        return ra_dec_point_src

    @staticmethod
    def find_iter_point_src(img, wcs, psf, channel, fwhm_point=2.0, thresh_point=3.0):

        src_catalog_point = VisualizeHelper.get_photutil_catalog(img=img, fwhm=fwhm_point, thresh=thresh_point)
        if len(src_catalog_point) == 0:
            pixel_coords = VisualizeHelper.get_brightest_pixel_coordinates(img=img)
            pixel_point_src = np.stack((pixel_coords[1], pixel_coords[0]), axis=1)
        else:
            pixel_point_src = np.stack((src_catalog_point['ycentroid'], src_catalog_point['xcentroid']), axis=1)
        obs, model_frame = VisualizeHelper.create_scarlet_obs(img=img, wcs=wcs, psf=psf, channel=channel)
        ra_dec_point_src = obs.get_sky_coord(pixel_point_src)

        # render the sources
        obs, model_frame, scarlet_sources =\
             VisualizeHelper.scarlet_fit_sources2img(obs=obs, model_frame=model_frame, ra_dec_point_src=ra_dec_point_src,
                                                     ra_dec_extended_src=None,
                                                     fitting_steps=100, fitting_rel_err=1e-18)

        # get residuals
        residuals = VisualizeHelper.get_scarlet_residuals(obs=obs, model_frame=model_frame, scarlet_sources=scarlet_sources)
        # plt.imshow(residuals[0])
        # plt.show()

        new_src_catalog_point = VisualizeHelper.get_photutil_catalog(img=residuals[0], fwhm=fwhm_point*20, thresh=thresh_point)
        if new_src_catalog_point is not None:
            new_pixel_point_src = np.stack((new_src_catalog_point['ycentroid'], new_src_catalog_point['xcentroid']), axis=1)
            new_ra_dec_point_src = obs.get_sky_coord(new_pixel_point_src)

            if ra_dec_point_src.ndim == 1:
                len_first = 1
            else:
                len_first = len(ra_dec_point_src)
            if new_ra_dec_point_src.ndim == 1:
                len_second = 1
            else:
                len_second = len(new_ra_dec_point_src)

            ra_dec = np.zeros((len_first + len_second, 2))
            ra_dec[0:len_first] = ra_dec_point_src
            ra_dec[len_first:] = new_ra_dec_point_src
        else:
            ra_dec = ra_dec_point_src
        return ra_dec

    @staticmethod
    def sep_source_extr(img_file, cutout_pos, cutout_size, obs, hdu_number, psf_file_name, chan_name,
                        deblend_nthresh, deblend_cont, aperture_scalings=None, snr_thresh=3,
                        flux_extraction_flag=False):

        if aperture_scalings is None:
            aperture_scalings = [1.0, 1.5, 2.0, 2.5, 3.0]
        if obs == 'hst':
            cutout = VisualizeHelper.get_hst_cutout_from_file(file_name=img_file, hdu_number=hdu_number,
                                                              cutout_pos=cutout_pos, cutout_size=cutout_size,
                                                              rescaling='mJy')
            psf_hdu = fits.open(psf_file_name)
            psf_data = VisualizeHelper.rebin_2d_array(psf_hdu[0].data[0:100, 0:100], (25, 25))
            # psf = scarlet.ImagePSF(psf_data)
        elif obs == 'jwst':
            cutout = VisualizeHelper.get_jwst_cutout_from_file(file_name=img_file, hdu_number=hdu_number,
                                                               cutout_pos=cutout_pos, cutout_size=cutout_size,
                                                               rescaling='mJy')
            psf_hdu = fits.open(psf_file_name)
            psf_data = VisualizeHelper.rebin_2d_array(psf_hdu[0].data[:-1, :-1],
                                                      (int(psf_hdu[0].data[:-1, :-1].shape[0]/4),
                                                       int(psf_hdu[0].data[:-1, :-1].shape[1]/4),))
            # psf = scarlet.ImagePSF(psf_data)
        else:
            raise KeyError('obs must be JWST or HST')

        # rescale the data. This is because Scarlet does not work with small number...
        # cutout.data *= rescaling_factor

        # m, s = np.mean(cutout.data), np.std(cutout.data)
        # plt.imshow(cutout.data, interpolation='nearest', cmap='gray', vmin=m-s, vmax=m+s, origin='lower')
        # plt.colorbar()
        # plt.show()
        bkg = sep.Background(np.array(cutout.data, dtype=float), bw=64, bh=64, fw=3, fh=3)
        # background of the image
        # bkg_image = bkg.back()
        # plt.imshow(bkg_image, interpolation='nearest', cmap='gray', origin='lower')
        # plt.colorbar()
        # plt.show()
        # get background noise
        bkg_rms = bkg.rms()
        # plt.imshow(bkg_rms, interpolation='nearest', cmap='gray', origin='lower')
        # plt.colorbar()
        # plt.show()

        # plt.imshow(psf_data, interpolation='nearest', cmap='gray', origin='lower')
        # plt.show()

        # remove background
        data_sub = cutout.data - bkg
        # So here we’re setting the detection threshold to be a constant value of 1.5sigma
        # where sigma is the global background RMS.
        objects = sep.extract(data_sub, snr_thresh, err=bkg.globalrms, filter_kernel=psf_data, minarea=5,
                              # deblend_nthresh=100, deblend_cont=0.00001)
                              deblend_nthresh=deblend_nthresh, deblend_cont=deblend_cont)

        # exclude sources near to the edge
        mask_good_app_points = ((objects['x'] > 1.0) & (objects['x'] < (data_sub.shape[0]) - 1.0) &
                                    (objects['y'] > 1.0) & (objects['y'] < (data_sub.shape[1]) - 1.0))
        objects = objects[mask_good_app_points]
        # data, thresh, err=None, mask=None, minarea=5,
        # filter_kernel=None, filter_type='matched',
        # deblend_nthresh=32, deblend_cont=0.005, clean=True, clean_param=1.0, segmentation_map=False):

        global_rms = bkg.globalrms

        # get the coordinates of the sources
        source_coords = cutout.wcs.pixel_to_world(objects['x'], objects['y'])
        # print('source_coords ', source_coords)
        source_dict = {'cutout_%s' % chan_name: cutout,
                       'global_rms_%s' % chan_name: global_rms,
                       'objects_%s' % chan_name: objects,
                       'source_coords_%s' % chan_name: source_coords}
        # compute source_fluxes with aperture scalings
        for scaling in aperture_scalings:
            # fig, ax = plt.subplots()
            # m, s = np.mean(data_sub), np.std(data_sub)
            # ax.imshow(data_sub, interpolation='nearest', cmap='gray',
            #                vmin=m-s, vmax=m+10*s)
            # # plot an ellipse for each object
            # for i in range(len(objects)):
            #     e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
            #                 width=objects['a'][i],
            #                 height=objects['b'][i],
            #                 angle=objects['theta'][i] * 180. / np.pi, alpha=0.7, linewidth=2)
            #     e.set_facecolor('none')
            #     e.set_edgecolor('c')
            #     ax.add_artist(e)
            # # plot an ellipse for each object
            # plt.show()

            flux, flux_err, flag = sep.sum_ellipse(data=data_sub, x=objects['x'], y=objects['y'],
                                                   a=objects['a'], b=objects['b'],
                                                   theta=objects['theta'], r=scaling, err=bkg_rms,
                                                   var=None, mask=None, maskthresh=0.0, seg_id=None, segmap=None,
                                                   bkgann=None, gain=None, subpix=5)
            source_dict.update({'flux_%.1f_%s' % (scaling, chan_name): flux,
                                'flux_err_%.1f_%s' % (scaling, chan_name): flux_err,
                                'flag_%.1f_%s' % (scaling, chan_name): flag})


        return source_dict

    @staticmethod
    def plot_sep_source_ext(cutout_dict, channel_list, circle_position, circle_radius_list):


        # build up a figure
        figure = plt.figure(figsize=(20, 10))

        ax_hst_f275w = figure.add_axes([-0.05, 0.67, 0.3, 0.3], projection=cutout_dict['cutout_%s' % channel_list[0]].wcs)
        ax_hst_f336w = figure.add_axes([0.12, 0.67, 0.3, 0.3], projection=cutout_dict['cutout_%s' % channel_list[1]].wcs)
        ax_hst_f438w = figure.add_axes([0.28, 0.67, 0.3, 0.3], projection=cutout_dict['cutout_%s' % channel_list[2]].wcs)
        ax_hst_f555w = figure.add_axes([0.5, 0.67, 0.3, 0.3], projection=cutout_dict['cutout_%s' % channel_list[3]].wcs)
        ax_hst_f814w = figure.add_axes([0.7, 0.67, 0.3, 0.3], projection=cutout_dict['cutout_%s' % channel_list[4]].wcs)

        ax_jwst_f200w = figure.add_axes([-0.05, 0.365, 0.3, 0.3], projection=cutout_dict['cutout_%s' % channel_list[5]].wcs)
        ax_jwst_f300m = figure.add_axes([0.12, 0.365, 0.3, 0.3], projection=cutout_dict['cutout_%s' % channel_list[6]].wcs)
        ax_jwst_f335m = figure.add_axes([0.28, 0.365, 0.3, 0.3], projection=cutout_dict['cutout_%s' % channel_list[7]].wcs)
        ax_jwst_f360m = figure.add_axes([0.5, 0.365, 0.3, 0.3], projection=cutout_dict['cutout_%s' % channel_list[8]].wcs)

        ax_jwst_f770w = figure.add_axes([-0.05, 0.06, 0.3, 0.3], projection=cutout_dict['cutout_%s' % channel_list[9]].wcs)
        ax_jwst_f1000w = figure.add_axes([0.12, 0.06, 0.3, 0.3], projection=cutout_dict['cutout_%s' % channel_list[10]].wcs)
        ax_jwst_f1130w = figure.add_axes([0.28, 0.06, 0.3, 0.3], projection=cutout_dict['cutout_%s' % channel_list[11]].wcs)
        ax_jwst_f2100w = figure.add_axes([0.5, 0.06, 0.3, 0.3], projection=cutout_dict['cutout_%s' % channel_list[12]].wcs)

        ax_list = [ax_hst_f275w, ax_hst_f336w, ax_hst_f438w, ax_hst_f555w, ax_hst_f814w,
                   ax_jwst_f200w, ax_jwst_f300m, ax_jwst_f335m, ax_jwst_f360m,
                   ax_jwst_f770w, ax_jwst_f1000w, ax_jwst_f1130w, ax_jwst_f2100w]

        for index in range(len(channel_list)):
            ax = ax_list[index]
            data = cutout_dict['cutout_%s' % channel_list[index]].data
            objects = cutout_dict['objects_%s' % channel_list[index]]
            # plot background-subtracted image
            m, s = np.mean(data), np.std(data)
            ax.imshow(data, interpolation='nearest', cmap='gray',
                           vmin=m-s, vmax=m+10*s)
            VisualizeHelper.erase_axis(ax=ax)
            VisualizeHelper.plot_coord_circle(ax=ax, position=circle_position, radius=circle_radius_list[index]*u.arcsec, color='r')
            # plot an ellipse for each object
            for i in range(len(objects)):
                e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
                            width=objects['a'][i],
                            height=objects['b'][i],
                            angle=objects['theta'][i] * 180. / np.pi, alpha=0.7, linewidth=2)
                e.set_facecolor('none')
                e.set_edgecolor('c')
                ax.add_artist(e)
            # plot an ellipse for each object
            for i in range(len(objects)):
                e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
                            width=2*objects['a'][i],
                            height=2*objects['b'][i],
                            angle=objects['theta'][i] * 180. / np.pi, alpha=0.7, linewidth=2)
                e.set_facecolor('none')
                e.set_edgecolor('c')
                ax.add_artist(e)

            # plot an ellipse for each object
            for i in range(len(objects)):
                e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
                            width=3*objects['a'][i],
                            height=3*objects['b'][i],
                            angle=objects['theta'][i] * 180. / np.pi, alpha=0.7, linewidth=2)
                e.set_facecolor('none')
                e.set_edgecolor('c')
                ax.add_artist(e)


        return figure

    @staticmethod
    def plot_sep_flux_ext(cutout_dict, channel_list, circle_position, circle_radius_list, scaling_list,
                          fontsize=18):

        # build up a figure
        figure = plt.figure(figsize=(18, 10))

        ax_hst_f275w = figure.add_axes([-0.05, 0.67, 0.285, 0.285], projection=cutout_dict['cutout_%s' % channel_list[0]].wcs)
        ax_hst_f336w = figure.add_axes([0.15, 0.67, 0.285, 0.285], projection=cutout_dict['cutout_%s' % channel_list[1]].wcs)
        ax_hst_f438w = figure.add_axes([0.35, 0.67, 0.285, 0.285], projection=cutout_dict['cutout_%s' % channel_list[2]].wcs)
        ax_hst_f555w = figure.add_axes([0.55, 0.67, 0.285, 0.285], projection=cutout_dict['cutout_%s' % channel_list[3]].wcs)
        ax_hst_f814w = figure.add_axes([0.75, 0.67, 0.285, 0.285], projection=cutout_dict['cutout_%s' % channel_list[4]].wcs)

        ax_jwst_f200w = figure.add_axes([-0.05, 0.34, 0.285, 0.285], projection=cutout_dict['cutout_%s' % channel_list[5]].wcs)
        ax_jwst_f300m = figure.add_axes([0.15, 0.34, 0.285, 0.285], projection=cutout_dict['cutout_%s' % channel_list[6]].wcs)
        ax_jwst_f335m = figure.add_axes([0.35, 0.34, 0.285, 0.285], projection=cutout_dict['cutout_%s' % channel_list[7]].wcs)
        ax_jwst_f360m = figure.add_axes([0.55, 0.34, 0.285, 0.285], projection=cutout_dict['cutout_%s' % channel_list[8]].wcs)

        ax_jwst_f770w = figure.add_axes([-0.05, 0.01, 0.285, 0.285], projection=cutout_dict['cutout_%s' % channel_list[9]].wcs)
        ax_jwst_f1000w = figure.add_axes([0.15, 0.01, 0.285, 0.285], projection=cutout_dict['cutout_%s' % channel_list[10]].wcs)
        ax_jwst_f1130w = figure.add_axes([0.35, 0.01, 0.285, 0.285], projection=cutout_dict['cutout_%s' % channel_list[11]].wcs)
        ax_jwst_f2100w = figure.add_axes([0.55, 0.01, 0.285, 0.285], projection=cutout_dict['cutout_%s' % channel_list[12]].wcs)

        ax_list = [ax_hst_f275w, ax_hst_f336w, ax_hst_f438w, ax_hst_f555w, ax_hst_f814w,
                   ax_jwst_f200w, ax_jwst_f300m, ax_jwst_f335m, ax_jwst_f360m,
                   ax_jwst_f770w, ax_jwst_f1000w, ax_jwst_f1130w, ax_jwst_f2100w]

        for index in range(len(channel_list)):
            ax = ax_list[index]
            data = cutout_dict['cutout_%s' % channel_list[index]].data
            objects = cutout_dict['objects_%s' % channel_list[index]]
            # plot background-subtracted image
            m, s = np.mean(data), np.std(data)
            ax.imshow(data, interpolation='nearest', cmap='gray',
                           vmin=m-s, vmax=m+10*s)
            VisualizeHelper.erase_axis(ax=ax)
            VisualizeHelper.plot_coord_circle(ax=ax, position=circle_position, radius=circle_radius_list[index]*u.arcsec, color='r')
            ax.set_title('%s' % channel_list[index], fontsize=fontsize)
            # plot an ellipse for each object
            source_positions = cutout_dict['source_coords_%s' % channel_list[index]]
            sep = source_positions.separation(circle_position)
            print(sep < circle_radius_list[index]*u.arcsec)
            cross_match_sources = sep < circle_radius_list[index]*u.arcsec

            for i in range(len(objects)):
                if cross_match_sources[i]:
                    e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
                                width=scaling_list[index] * objects['a'][i],
                                height=scaling_list[index] * objects['b'][i],
                                angle=objects['theta'][i] * 180. / np.pi, alpha=0.7, linewidth=2)
                    e.set_facecolor('none')
                    e.set_edgecolor('c')
                    ax.add_artist(e)

        return figure

    @staticmethod
    def get_sep_source_fluxes(cutout_dict, channel_list, circle_position, circle_radius_list, scaling_list):

        flux_dict = {}

        for index in range(len(channel_list)):
            source_positions = cutout_dict['source_coords_%s' % channel_list[index]]
            # print('source_positions ', source_positions)
            # ra_circ = circle_position.ra
            # dec_circ = circle_position.dec
            # print('source_positions ra', source_positions.ra)
            # print('ra_circ ', ra_circ)
            # print('dec_circ ', dec_circ)
            # diff = source_positions.ra - ra_circ
            # print('diff ', diff)
            # print(diff < circle_radius_list[index])
            sep = source_positions.separation(circle_position)
            print(sep < circle_radius_list[index]*u.arcsec)
            # exit()
            cross_match_sources = sep < circle_radius_list[index]*u.arcsec

            if sum(cross_match_sources) == 0:
                flux_dict.update({'flux_%s' % channel_list[index]: cutout_dict['global_rms_%s' % channel_list[index]]})
                flux_dict.update({'flux_err_%s' % channel_list[index]: -1.0})
            else:
                flux_dict.update({'flux_%s' % channel_list[index]:
                                      cutout_dict['flux_%.1f_%s' % (scaling_list[index], channel_list[index])][cross_match_sources]
                                  })
                flux_dict.update({'flux_err_%s' % channel_list[index]:
                                      cutout_dict['flux_err_%.1f_%s' % (scaling_list[index], channel_list[index])][cross_match_sources]
                                  })

        return flux_dict
    @staticmethod
    def find_point_and_ext_src(img, wcs, psf, channel,
                              fwhm_point=2.0, thresh_point=5.0, fwhm_ext=5.0, thresh_ext=3.0):

        # get point sources
        ra_dec_point_src = VisualizeHelper.find_point_src(img=img, wcs=wcs, psf=psf, channel=channel,
                                                          fwhm_point=fwhm_point, thresh_point=thresh_point)

        obs, model_frame = VisualizeHelper.create_scarlet_obs(img=img, wcs=wcs, psf=psf, channel=channel)

        obs, model_frame, scarlet_sources = VisualizeHelper.scarlet_fit_sources2img(obs=obs, model_frame=model_frame,
                                                                                    ra_dec_point_src=
                                                                                    ra_dec_point_src,
                                                                                    ra_dec_extended_src=None)
        residual = VisualizeHelper.get_scarlet_residuals(obs=obs, model_frame=model_frame,
                                                         scarlet_sources=scarlet_sources)
        ra_dec_ext_src = VisualizeHelper.find_point_src(img=residual[0], wcs=wcs, psf=psf, channel=channel,
                                                        fwhm_point=fwhm_ext, thresh_point=thresh_ext)

        return ra_dec_point_src, ra_dec_ext_src

    @staticmethod
    def indi_point_fit2cutout(img_file, cutout_pos, cutout_size, obs, hdu_number, psf_file_name, chan_name,
                              fwhm_point=1, thresh_point=3,
                              fitting_steps=50, fitting_rel_err=1e-16, rescaling_factor=1e6):

        if obs == 'hst':
            cutout = VisualizeHelper.get_hst_cutout_from_file(file_name=img_file, hdu_number=hdu_number,
                                                              cutout_pos=cutout_pos, cutout_size=cutout_size,
                                                              rescaling='mJy')
            psf_hdu = fits.open(psf_file_name)
            psf_data = VisualizeHelper.rebin_2d_array(psf_hdu[0].data[0:100, 0:100], (25, 25))
            psf = scarlet.ImagePSF(psf_data)
        elif obs == 'jwst':
            cutout = VisualizeHelper.get_jwst_cutout_from_file(file_name=img_file, hdu_number=hdu_number,
                                                               cutout_pos=cutout_pos, cutout_size=cutout_size,
                                                               rescaling='mJy')
            psf_hdu = fits.open(psf_file_name)
            psf_data = VisualizeHelper.rebin_2d_array(psf_hdu[0].data[:-1, :-1],
                                                      (int(psf_hdu[0].data[:-1, :-1].shape[0]/4),
                                                       int(psf_hdu[0].data[:-1, :-1].shape[1]/4),))
            psf = scarlet.ImagePSF(psf_data)
        else:
            raise KeyError('obs must be JWST or HST')

        # rescale the data. This is because Scarlet does not work with small number...
        cutout.data *= rescaling_factor

        ra_dec_point_src = \
            VisualizeHelper.find_iter_point_src(img=cutout.data, wcs=cutout.wcs, psf=psf, channel=[chan_name],
                                                fwhm_point=fwhm_point, thresh_point=thresh_point)
        # print('ra_dec_point_src ', ra_dec_point_src)

        obs, model_frame = VisualizeHelper.create_scarlet_obs(img=cutout.data, wcs=cutout.wcs, psf=psf,
                                                              channel=[chan_name])

        # scarlet_sources = VisualizeHelper.init_sources(obs=obs, model_frame=model_frame,
        #                                                ra_dec_point_src=ra_dec_point_src)

        obs, model_frame, scarlet_sources = VisualizeHelper.scarlet_fit_sources2img(obs=obs,
                                                                                    model_frame=model_frame,
                                                                                    ra_dec_point_src=
                                                                                    ra_dec_point_src,
                                                                                    ra_dec_extended_src=
                                                                                    None,
                                                                                    fitting_steps=fitting_steps,
                                                                                    fitting_rel_err=fitting_rel_err)
        cutout_dict = {'ra_dec_point_src_%s' % chan_name: ra_dec_point_src,
                       'obs_%s' % chan_name: obs,
                       'model_frame_%s' % chan_name: model_frame,
                       'scarlet_sources_%s' % chan_name: scarlet_sources,
                       'cutout_%s' % chan_name: cutout}
        return cutout_dict

    @staticmethod
    def indi_ext_fit2cutout(img_file, cutout_pos, cutout_size, obs, hdu_number, psf_file_name, chan_name,
                            fwhm_point=1, thresh_point=3, box_size_extended_arc=0.1,
                            fitting_steps=50, fitting_rel_err=1e-16, rescaling_factor=1e6):

        if obs == 'hst':
            cutout = VisualizeHelper.get_hst_cutout_from_file(file_name=img_file, hdu_number=hdu_number,
                                                              cutout_pos=cutout_pos, cutout_size=cutout_size,
                                                              rescaling='mJy')
            psf_hdu = fits.open(psf_file_name)
            psf_data = VisualizeHelper.rebin_2d_array(psf_hdu[0].data[0:100, 0:100], (25, 25))
            psf = scarlet.ImagePSF(psf_data)
        elif obs == 'jwst':
            cutout = VisualizeHelper.get_jwst_cutout_from_file(file_name=img_file, hdu_number=hdu_number,
                                                               cutout_pos=cutout_pos, cutout_size=cutout_size,
                                                               rescaling='mJy')
            psf_hdu = fits.open(psf_file_name)
            psf_data = VisualizeHelper.rebin_2d_array(psf_hdu[0].data[:-1, :-1],
                                                      (int(psf_hdu[0].data[:-1, :-1].shape[0]/4),
                                                       int(psf_hdu[0].data[:-1, :-1].shape[1]/4),))
            psf = scarlet.ImagePSF(psf_data)
        else:
            raise KeyError('obs must be JWST or HST')

        # rescale the data. This is because Scarlet does not work with small number...
        cutout.data *= rescaling_factor

        ra_dec_point_src = \
            VisualizeHelper.find_iter_point_src(img=cutout.data, wcs=cutout.wcs, psf=psf, channel=[chan_name],
                                                fwhm_point=fwhm_point, thresh_point=thresh_point)
        print('ra_dec_point_src ', ra_dec_point_src)

        box_size_extended_ref = int(box_size_extended_arc / 3600 / proj_plane_pixel_scales(cutout.wcs)[0])
        print('box_size_extended_ref ', box_size_extended_ref)

        obs, model_frame = VisualizeHelper.create_scarlet_obs(img=cutout.data, wcs=cutout.wcs, psf=psf,
                                                              channel=[chan_name])

        obs, model_frame, scarlet_sources = VisualizeHelper.scarlet_fit_sources2img(obs=obs,
                                                                                    model_frame=model_frame,
                                                                                    ra_dec_point_src=
                                                                                    None,
                                                                                    ra_dec_extended_src=
                                                                                    ra_dec_point_src,
                                                                                    box_size_extended=box_size_extended_ref,
                                                                                    fitting_steps=fitting_steps,
                                                                                    fitting_rel_err=fitting_rel_err)
        cutout_dict = {'ra_dec_point_src_%s' % chan_name: ra_dec_point_src,
                       'obs_%s' % chan_name: obs,
                       'model_frame_%s' % chan_name: model_frame,
                       'scarlet_sources_%s' % chan_name: scarlet_sources,
                       'cutout_%s' % chan_name: cutout}
        return cutout_dict

    @staticmethod
    def plot_scarlet_results(ax_model, ax_data, ax_residuals, obs, model_frame, scarlet_sources, channel,
                             x_labels_model=True, x_labels_data=True, x_labels_residuals=True,
                             y_labels_model=True, y_labels_data=True, y_labels_residuals=True, fontsize=20,
                             ra_circle=None, dec_circle=None, circle_rad=0.5, show_sources=False):

        model = VisualizeHelper.get_scarlet_model(obs=obs, model_frame=model_frame, scarlet_sources=scarlet_sources)
        residuals = VisualizeHelper.get_scarlet_residuals(obs=obs, model_frame=model_frame,
                                                          scarlet_sources=scarlet_sources)
        obs_data = obs.data

        mean, median, std = sigma_clipped_stats(obs_data, sigma=6.0)

        # plot galaxy overview image
        ax_model.set_title('Model ' + channel, fontsize=20)
        ax_data.set_title('Observations ' + channel, fontsize=20)
        ax_residuals.set_title('Residuals ' + channel, fontsize=20)

        ax_model.imshow(model[0], vmin=-std, vmax=10*std, cmap='Greys')
        ax_data.imshow(obs_data[0], vmin=-std, vmax=10*std, cmap='Greys')
        ax_residuals.imshow(residuals[0], vmin=-3*std, vmax=3*std, cmap='Greys')

        if show_sources:
            for src in scarlet_sources:
                if isinstance(src, scarlet.source.PointSource):
                    color = 'r'
                elif (isinstance(src, scarlet.source.SingleExtendedSource) |
                      isinstance(src, scarlet.source.CompactExtendedSource)):
                    color = 'b'
                else:
                    raise TypeError(' source is not known ? please implement this')
                ax_model.scatter(scarlet.measure.centroid(src)[2], scarlet.measure.centroid(src)[1], color=color)
                ax_data.scatter(scarlet.measure.centroid(src)[2], scarlet.measure.centroid(src)[1], color=color)
                ax_residuals.scatter(scarlet.measure.centroid(src)[2], scarlet.measure.centroid(src)[1], color=color)


        ax_model.tick_params(axis='both', which='both', width=1.5, direction='in', color='k', labelsize=fontsize)
        ax_data.tick_params(axis='both', which='both', width=1.5, direction='in', color='k', labelsize=fontsize)
        ax_residuals.tick_params(axis='both', which='both', width=1.5, direction='in', color='k', labelsize=fontsize)
        ax_model.coords['ra'].display_minor_ticks(True)
        ax_model.coords['dec'].display_minor_ticks(True)
        ax_data.coords['ra'].display_minor_ticks(True)
        ax_data.coords['dec'].display_minor_ticks(True)
        ax_residuals.coords['ra'].display_minor_ticks(True)
        ax_residuals.coords['dec'].display_minor_ticks(True)

        if x_labels_model:
            ax_model.coords['ra'].set_axislabel('R.A. (2000.0)', minpad=1, fontsize=fontsize)
        else:
            ax_model.coords['ra'].set_ticklabel_visible(False)
            ax_model.coords['ra'].set_axislabel(' ')
        if y_labels_model:
            ax_model.coords['dec'].set_ticklabel(rotation=60)
            ax_model.coords['dec'].set_axislabel('DEC. (2000.0)', minpad=0, fontsize=fontsize)
        else:
            ax_model.coords['dec'].set_ticklabel_visible(False)
            ax_model.coords['dec'].set_axislabel(' ')

        if x_labels_data:
            ax_data.coords['ra'].set_axislabel('R.A. (2000.0)', minpad=1, fontsize=fontsize)
        else:
            ax_data.coords['ra'].set_ticklabel_visible(False)
            ax_data.coords['ra'].set_axislabel(' ')
        if y_labels_data:
            ax_data.coords['dec'].set_ticklabel(rotation=60)
            ax_data.coords['dec'].set_axislabel('DEC. (2000.0)', minpad=0, fontsize=fontsize)
        else:
            ax_data.coords['dec'].set_ticklabel_visible(False)
            ax_data.coords['dec'].set_axislabel(' ')

        if x_labels_residuals:
            ax_residuals.coords['ra'].set_axislabel('R.A. (2000.0)', minpad=1, fontsize=fontsize)
        else:
            ax_residuals.coords['ra'].set_ticklabel_visible(False)
            ax_residuals.coords['ra'].set_axislabel(' ')
        if y_labels_residuals:
            ax_residuals.coords['dec'].set_ticklabel(rotation=60)
            ax_residuals.coords['dec'].set_axislabel('DEC. (2000.0)', minpad=0, fontsize=fontsize)
        else:
            ax_residuals.coords['dec'].set_ticklabel_visible(False)
            ax_residuals.coords['dec'].set_axislabel(' ')

        if (ra_circle is not None) & (dec_circle is not None):

            circle_coords = SkyCoord(ra=ra_circle, dec=dec_circle, unit=(u.degree, u.degree), frame='icrs')
            VisualizeHelper.plot_coord_circle(ax=ax_model, position=circle_coords,
                                              radius=circle_rad * u.arcsec, color='r',
                                              linestyle='-', linewidth=2, alpha=1.)
            VisualizeHelper.plot_coord_circle(ax=ax_data, position=circle_coords,
                                              radius=circle_rad * u.arcsec, color='r',
                                              linestyle='-', linewidth=2, alpha=1.)
            VisualizeHelper.plot_coord_circle(ax=ax_residuals, position=circle_coords,
                                              radius=circle_rad * u.arcsec, color='r',
                                              linestyle='-', linewidth=2, alpha=1.)



