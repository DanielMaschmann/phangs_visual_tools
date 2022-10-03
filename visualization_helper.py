import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c as speed_of_light
import matplotlib.image as mpimg

import astropy.units as u
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.io import fits


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
            vmax_hst = np.max([np.max(cutout_hst_f275w.data), np.max(cutout_hst_f438w.data),
                               np.max(cutout_hst_f555w.data), np.max(cutout_hst_f814w.data)])
        if vmax_nircam is None:
            vmax_nircam = np.max([np.max(cutout_jwst_f200w.data), np.max(cutout_jwst_f300m.data),
                                  np.max(cutout_jwst_f335m.data), np.max(cutout_jwst_f360m.data)])
        if vmax_miri is None:
            vmax_miri = np.max([np.max(cutout_jwst_f770w.data), np.max(cutout_jwst_f1000w.data),
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
            cb_hst = ax_hst_f275w.imshow(np.log10(cutout_hst_f275w.data), vmin=np.log10(vmax_hst/100),
                                         vmax=np.log10(vmax_hst), cmap=cmap_hst)
            ax_hst_f438w.imshow(np.log10(cutout_hst_f438w.data), vmin=np.log10(vmax_hst/100), vmax=np.log10(vmax_hst),
                                cmap=cmap_hst)
            ax_hst_f555w.imshow(np.log10(cutout_hst_f555w.data), vmin=np.log10(vmax_hst/100), vmax=np.log10(vmax_hst),
                                cmap=cmap_hst)
            ax_hst_f814w.imshow(np.log10(cutout_hst_f814w.data), vmin=np.log10(vmax_hst/100), vmax=np.log10(vmax_hst),
                                cmap=cmap_hst)

            cb_nircam = ax_jwst_f200w.imshow(np.log10(cutout_jwst_f200w.data), vmin=np.log10(vmax_nircam/100),
                                             vmax=np.log10(vmax_nircam), cmap=cmap_nircam)
            ax_jwst_f300m.imshow(np.log10(cutout_jwst_f300m.data), vmin=np.log10(vmax_nircam/100),
                                 vmax=np.log10(vmax_nircam), cmap=cmap_nircam)
            ax_jwst_f335m.imshow(np.log10(cutout_jwst_f335m.data), vmin=np.log10(vmax_nircam/100),
                                 vmax=np.log10(vmax_nircam), cmap=cmap_nircam)
            ax_jwst_f360m.imshow(np.log10(cutout_jwst_f360m.data), vmin=np.log10(vmax_nircam/100),
                                 vmax=np.log10(vmax_nircam), cmap=cmap_nircam)

            cb_miri = ax_jwst_f770w.imshow(np.log10(cutout_jwst_f770w.data), vmin=np.log10(vmax_miri/100),
                                           vmax=np.log10(vmax_miri), cmap=cmap_miri)
            ax_jwst_f1000w.imshow(np.log10(cutout_jwst_f1000w.data), vmin=np.log10(vmax_miri/100),
                                  vmax=np.log10(vmax_miri), cmap=cmap_miri)
            ax_jwst_f1130w.imshow(np.log10(cutout_jwst_f1130w.data), vmin=np.log10(vmax_miri/100),
                                  vmax=np.log10(vmax_miri), cmap=cmap_miri)
            ax_jwst_f2100w.imshow(np.log10(cutout_jwst_f2100w.data), vmin=np.log10(vmax_miri/100),
                                  vmax=np.log10(vmax_miri), cmap=cmap_miri)
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
            cigale_flux_file_col_name_ord = np.array(['id', 'redshift',
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
        # for names in hdu_best_model[1].header.keys():
        #     print(names, hdu_best_model[1].header[names])

        results = fits.open(flux_fitted_model_file_name)
        # print(results.info())
        results_data = results[1].data

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

        ax_fit.set_ylabel(r'S$_{\nu}$ (mJy)', fontsize=fontsize)
        best_fit_residuals.set_xlabel(r'Observed ${\lambda}$ ($\mu$m)', labelpad=-2, fontsize=fontsize)
        best_fit_residuals.set_ylabel('  Relative \n residuals', labelpad=-9, fontsize=fontsize - 2)
        best_fit_residuals.text(5e3, 0.4, '(Obs - Mod) / Obs', color='k', fontsize=fontsize - 4)
        ax_fit.legend(frameon=False, fontsize=fontsize-6, bbox_to_anchor=[0.5, 0.5])
        ax_fit.text(3.14, 0.002, 'stellar age = %i Myr' % int(hdu_best_model[1].header['sfh.age']), fontsize=fontsize-6)
        ax_fit.text(3.14, 0.0003, 'E(B-V) = %.1f' % float(hdu_best_model[1].header['attenuation.E_BV']),
                    fontsize=fontsize-6)
        ax_fit.text(3.14, 0.00005, 'stellar met = %.2f' % float(hdu_best_model[1].header['stellar.metallicity']),
                    fontsize=fontsize-6)
        best_fit_residuals.text(13.5e0, -0.5, '(Obs - Mod) / Obs', color='k', fontsize=fontsize - 4)
        ax_fit.tick_params(axis='both', which='both', width=4, length=5, direction='in', pad=10, labelsize=fontsize)
        best_fit_residuals.tick_params(axis='both', which='both', width=4, length=5, top=True, direction='in', labelsize=fontsize-4)
        best_fit_residuals.tick_params(axis='x', which='major', pad=10)
        ax_fit.set_xlim(200 * 1e-3, 3e1)
        best_fit_residuals.set_xlim(200 * 1e-3, 3e1)
        ax_fit.set_ylim(0.000009, 5e7)
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