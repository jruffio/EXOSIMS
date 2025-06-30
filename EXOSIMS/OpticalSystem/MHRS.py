# -*- coding: utf-8 -*-
import warnings

import astropy.units as u
import numpy as np
from scipy.optimize import minimize, root_scalar
from  scipy.interpolate import interp1d
from tqdm import tqdm

from EXOSIMS.Prototypes.OpticalSystem import OpticalSystem
from EXOSIMS.util._numpy_compat import copy_if_needed


from synphot import Observation
from synphot import SourceSpectrum
import synphot
from astropy.modeling.models import Tabular1D
from joblib import Parallel, delayed
from astropy.units import Quantity

class MHRS(OpticalSystem):
    r"""Optical System class for moderate to high resolution spectroscopy (MHRS)

    Note: File modified from OpticalSystem.Nemati.py.

    Args:
        CIC (float):
            Default clock-induced-charge (in electrons/pixel/read).  Only used
            when not set in science instrument definition. Defaults to 1e-3
        radDos (float):
            Default radiation dose.   Only used when not set in mode definition.
            Specific defintion depends on particular optical system. Defaults to 0.
        PCeff (float):
            Default photon counting efficiency.  Only used when not set
            in science instrument definition. Defaults to 0.8
        ENF (float):
            Default excess noise factor.  Only used when not set
            in science instrument definition. Defaults to 1.
        ref_dMag (float):
            Reference star :math:`\Delta\mathrm{mag}` for reference differential
            imaging.  Defaults to 3.  Unused if ``ref_Time`` input is 0
        ref_Time (float):
            Faction of time used on reference star imaging. Must be between 0 and 1.
            Defaults to 0
        **specs:
            :ref:`sec:inputspec`

    Attributes:
        default_vals_extra (dict):
            Dictionary of input values to be filled in as defaults in the instrument,
            starlight supporession system and observing modes. These values are specific
            to this module.
        ref_dMag (float):
            Reference star :math:`\Delta\mathrm{mag}` for reference differential
            imaging. Unused if ``ref_Time`` input is 0
        ref_Time (float):
            Faction of time used on reference star imaging.

    """

    def __init__(
        self, CIC=1e-3, radDos=0, PCeff=0.8, ENF=1, ref_dMag=3, ref_Time=0, **specs
    ):
        self.ref_dMag = float(ref_dMag)  # reference star dMag for RDI
        self.ref_Time = float(ref_Time)  # fraction of time spent on ref star for RDI

        # package inputs for use in popoulate*_extra
        self.default_vals_extra = {
            "CIC": CIC,
            "radDos": radDos,
            "PCeff": PCeff,
            "ENF": ENF,
        }

        # call upstream init
        OpticalSystem.__init__(self, **specs)

        # add local defaults to outspec
        for k in self.default_vals_extra:
            self._outspec[k] = self.default_vals_extra[k]

    def populate_scienceInstruments_extra(self):
        """Add Nemati-specific keywords to scienceInstruments"""
        newatts = [
            "CIC",  # clock-induced-charge
            "ENF",  # excess noise factor
            "PCeff",  # photon counting efficiency
        ]
        self.allowed_scienceInstrument_kws += newatts

        for ninst, inst in enumerate(self.scienceInstruments):
            for att in newatts:
                inst[att] = float(inst.get(att, self.default_vals_extra[att]))
                self._outspec["scienceInstruments"][ninst][att] = inst[att]

    def populate_observingModes_extra(self):
        """Add Nemati-specific observing mode keywords"""

        self.allowed_observingMode_kws.append("radDos")

        for nmode, mode in enumerate(self.observingModes):
            # radiation dosage, goes from 0 (beginning of mission) to 1 (end of mission)
            mode["radDos"] = float(
                mode.get("radDos", self.default_vals_extra["radDos"])
            )
            self._outspec["observingModes"][nmode]["radDos"] = mode["radDos"]

    # def Cp_Cb_Csp(self, TL, sInds, fZ, JEZ, dMag, WA, mode, returnExtra=False, TK=None):
    #     if "imager" in mode["inst"]["name"].lower():
    #         return self.Cp_Cb_Csp_imager(TL, sInds, fZ, JEZ, dMag, WA, mode, returnExtra=returnExtra, TK=TK)
    #     if "spectro" in mode["inst"]["name"].lower():
    #         return self.Cp_Cb_Csp_spectro(TL, sInds, fZ, JEZ, dMag, WA, mode, returnExtra=returnExtra, TK=TK)

    def Cp_Cb_Csp_helper(self, TL, sInds, fZ, JEZ, dMag, WA, mode):
        out = super().Cp_Cb_Csp_helper(TL, sInds, fZ, JEZ, dMag, WA, mode)
        #C_star, C_p, C_sr, C_z, C_ez, C_dc, C_bl, Npix = out

        if "spectro" in mode["inst"]["name"].lower():
            # Undo the flux scaling for spectroscopy mode:
            #     Why? By default, the flux is computed in a single spectral resolution element for spectroscopy, but that's not the desire
            # behavior since the S/N per bin is not the relevant metric here, so we make sure that the flux rates are
            # always computed for the full bandpass.
            spectral_bin_to_bandpass = 1/(mode["deltaLam_eff"]/ mode["deltaLam"])
            Npix = out[7]/mode["inst"]["lenslSamp"] # This is N pixels per bin
            C_dc = Npix * mode["inst"]["idark"] # This is dark current per bin
            C_star, C_p, C_sr, C_z, C_ez, C_bl = [f*spectral_bin_to_bandpass for f in out[0:5]+out[6:7]]
        else:
            C_star, C_p, C_sr, C_z, C_ez, C_dc, C_bl, Npix = out

        return C_star, C_p, C_sr, C_z, C_ez, C_dc, C_bl, Npix

    def Cp_Cb_Csp(self, TL, sInds, fZ, JEZ, dMag, WA, mode, returnExtra=False, TK=None):
        """Calculates electron count rates for planet signal, background noise,
        and speckle residuals.

        Args:
            TL (:ref:`TargetList`):
                TargetList class object
            sInds (~numpy.ndarray(int)):
                Integer indices of the stars of interest
            fZ (~astropy.units.Quantity(~numpy.ndarray(float))):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            JEZ (astropy Quantity array):
                Intensity of exo-zodiacal light in units of ph/s/m2/arcsec2
            dMag (~numpy.ndarray(float)):
                Differences in magnitude between planets and their host star
            WA (~astropy.units.Quantity(~numpy.ndarray(float))):
                Working angles of the planets of interest in units of arcsec
            mode (dict):
                Selected observing mode
            returnExtra (bool):
                Optional flag, default False, set True to return additional rates for
                validation
            TK (:ref:`TimeKeeping`, optional):
                Optional TimeKeeping object (default None), used to model detector
                degradation effects where applicable.


        Returns:
            tuple:
                C_p (~astropy.units.Quantity(~numpy.ndarray(float))):
                    Planet signal electron count rate in units of 1/s
                C_b (~astropy.units.Quantity(~numpy.ndarray(float))):
                    Background noise electron count rate in units of 1/s
                C_sp (~astropy.units.Quantity(~numpy.ndarray(float))):
                    Residual speckle spatial structure (systematic error)
                    in units of 1/s

        """
        # grab all count rates
        C_star, C_p0, C_sr, C_z, C_ez, C_dc, C_bl, Npix = self.Cp_Cb_Csp_helper(
            TL, sInds, fZ, JEZ, dMag, WA, mode
        )

        # Strip units for computation speed
        # Underscores are used to indicate that the variable is unitless and
        # need to have units added back before returning
        _C_star = C_star.to_value(self.inv_s)
        _C_p0 = C_p0.to_value(self.inv_s)
        _C_sr = C_sr.to_value(self.inv_s)
        _C_z = C_z.to_value(self.inv_s)
        _C_ez = C_ez.to_value(self.inv_s)
        _C_dc = C_dc.to_value(self.inv_s)
        _C_bl = C_bl.to_value(self.inv_s)
        inst = mode["inst"]

        # exposure time
        if self.texp_flag:
            with np.errstate(divide="ignore", invalid="ignore"):
                texp = 1 / _C_p0 / 10  # Use 1/C_p0 as frame time for photon counting
        else:
            texp = inst["texp"].to_value(u.s)
        # readout noise
        _C_rn = Npix * inst["sread"] / texp

        # clock-induced-charge
        _C_cc = Npix * inst["CIC"] / texp

        # C_p = PLANET SIGNAL RATE
        # photon counting efficiency
        PCeff = inst["PCeff"]
        # radiation dosage
        radDos = mode["radDos"]
        # photon-converted 1 frame (minimum 1 photon)
        # there may be zeros in the denominator. Suppress the resulting warning:
        with np.errstate(divide="ignore", invalid="ignore"):
            phConv = np.clip(((_C_p0 + _C_sr + _C_z + _C_ez) / Npix * texp), 1, None)
        # net charge transfer efficiency
        with np.errstate(invalid="ignore"):
            NCTE = 1.0 + (radDos / 4.0) * 0.51296 * (np.log10(phConv) + 0.0147233)
        # planet signal rate
        _C_p = _C_p0 * PCeff * NCTE
        # possibility of Npix=0 may lead C_p to be nan.  Change these to zero instead.
        _C_p[np.isnan(_C_p)] = 0

        # C_b = NOISE VARIANCE RATE
        # corrections for Ref star Differential Imaging e.g. dMag=3 and 20% time on ref
        # k_SZ for speckle and zodi light, and k_det for detector
        k_SZ = (
            1.0 + 1.0 / (10 ** (0.4 * self.ref_dMag) * self.ref_Time)
            if self.ref_Time > 0
            else 1.0
        )
        k_det = 1.0 + self.ref_Time
        # calculate Cb
        ENF2 = inst["ENF"] ** 2
        _C_b = k_SZ * ENF2 * (_C_sr + _C_z + _C_ez + _C_bl) + k_det * (
            ENF2 * (_C_dc + _C_cc) + _C_rn
        )
        # for characterization, Cb must include the planet
        if not (mode["detectionMode"]):
            _C_b = _C_b + ENF2 * _C_p0
            _C_sp = _C_sr * TL.PostProcessing.ppFact_char(WA) * self.stabilityFact
        else:
            # C_sp = spatial structure to the speckle including post-processing
            #        contrast factor and stability factor
            _C_sp = _C_sr * TL.PostProcessing.ppFact(WA) * self.stabilityFact

        if returnExtra:
            # organize components into an optional fourth result
            C_extra = dict(
                C_sr=_C_sr << self.inv_s,
                C_z=_C_z << self.inv_s,
                C_ez=_C_ez << self.inv_s,
                C_dc=_C_dc << self.inv_s,
                C_cc=_C_cc << self.inv_s,
                C_rn=_C_rn << self.inv_s,
                C_star=_C_star << self.inv_s,
                C_p0=_C_p0 << self.inv_s,
                C_bl=_C_bl << self.inv_s,
                Npix=Npix,
            )
            return _C_p << self.inv_s, _C_b << self.inv_s, _C_sp << self.inv_s, C_extra
        else:
            return _C_p << self.inv_s, _C_b << self.inv_s, _C_sp << self.inv_s

    def Cp_Cb_Csp_spec(self, TL, sInds, fZ, JEZ, dMag, WA, mode, returnExtra=False, TK=None, pl_waves = None,
                       pl_template = None, R_pl_template=None,pl_template_name=None,n_jobs=-1,broaden_pixel=True):
        """Similar to self.Cp_Cb_Csp() but returning spectra instead of broadband fluxes.
        Calculates different spectra with the electron count rates for planet signal, background noise,and speckle residuals.

        Args:
            TL (:ref:`TargetList`):
                TargetList class object
            sInds (~numpy.ndarray(int)):
                Integer indices of the stars of interest
            fZ (~astropy.units.Quantity(~numpy.ndarray(float))):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            JEZ (astropy Quantity array):
                Intensity of exo-zodiacal light in units of ph/s/m2/arcsec2
            dMag (~numpy.ndarray(float)):
                Differences in magnitude between planets and their host star
            WA (~astropy.units.Quantity(~numpy.ndarray(float))):
                Working angles of the planets of interest in units of arcsec
            mode (dict):
                Selected observing mode
            returnExtra (bool):
                Optional flag, default False, set True to return additional rates for
                validation
            TK (:ref:`TimeKeeping`, optional):
                Optional TimeKeeping object (default None), used to model detector
                degradation effects where applicable.
            pl_waves (~numpy.ndarray(float)):
                Wavelength array of planet spectral tempalte
            pl_template (List of ~numpy.ndarray(float)):
                List of planet albeda spectral template. If more than one, this can include molecular templates.
            R_pl_template:
                Spectral resolution of the pl_template spectrum as way to know what is the maximum spectral resolution for this calculation.
            pl_template_name (List of str)
                List of the names for the spectral templates in pl_template.
                e.g. ["all","H2O","O2"]
            n_jobs (int):
                Number of parallel jobs (-1 = all cores).
                Not parallelized if 0.
            broaden_pixel (Boolean):
                If True, subsequently broadens the spectrum the insturment resolution and then to the pixel width. Otherwise,
                only broaden to the instrumental resolution, and effectively assume that the pixel broadening is included in it.
                If samples_only is not None, having broaden_pixel=True is much slower.


        Returns:
            tuple:
                C_p (~astropy.units.Quantity(~numpy.ndarray(float))):
                    Planet signal electron count rate in units of 1/s
                C_b (~astropy.units.Quantity(~numpy.ndarray(float))):
                    Background noise electron count rate in units of 1/s
                C_sp (~astropy.units.Quantity(~numpy.ndarray(float))):
                    Residual speckle spatial structure (systematic error)
                    in units of 1/s

        """
        inst = mode["inst"]
        if "spectro" not in inst["name"].lower():
            raise Exception(inst["name"] + " is not a spectrograph. A spectrograph is needed for using Cp_Cb_Csp_spec().")

        # todo: do not hard code R_star_template = 500
        # todo: implement high res stellar models
        # todo: fix pl_template_incl_star = pl_template_cropped#*star_template_resampled
        # todo: make flat spectrum if pl_template is None

        if isinstance(pl_template, (np.ndarray)):
            pl_template = [pl_template]
        if pl_template_name is None:
            pl_template_name = "TBD"
        if isinstance(pl_template_name, str):
            pl_template_name = [pl_template_name]

        # Create output lists to manage the fact that a set of stars/WA/etc can be given as an input
        star_template_scaled_C_sr_list = []
        _C_z_spec_list = []
        _C_ez_spec_list = []
        _C_dc_spec_list = []
        _C_cc_spec_list = []
        _C_rn_spec_list = []
        _C_star_spec_list = []
        pl0_template_scaled_C_p0_list = []
        _C_bl_spec_list = []

        pl0_template_scaled_C_p_list = []
        _C_b_spec_list = []
        star_template_scaled_C_sp_list = []

        pl_mol_template_scaled_C_p_list = []

        # Assume `bandpass` is your synphot.SpectralElement object
        bandpass_waves = mode["bandpass"].waveset
        bandpass_filter = mode["bandpass"](bandpass_waves)
        bandpass_func = interp1d(bandpass_waves, bandpass_filter, bounds_error=False, fill_value=0)
        nonzero = bandpass_filter > 0.01
        min_wave_bandpass = bandpass_waves[nonzero][0]  # wavelength has units (Angstrom most likely)
        max_wave_bandpass = bandpass_waves[nonzero][-1]
        lambda_center = 0.5 * (min_wave_bandpass + max_wave_bandpass)

        if inst["Rs"] > R_pl_template / 2.:
            raise ValueError(
                "Instrument resolution is higher than 1/2 the planet template resolution.")

        # extract relevant subset of the template wavelength axes to speed up subsequent processing
        # Apply margin
        wmin_with_margin = min_wave_bandpass - 2 * min_wave_bandpass / inst["Rs"]
        wmax_with_margin = max_wave_bandpass + 2 * max_wave_bandpass / inst["Rs"]
        # crop planet template
        pl_mask = (pl_waves.to(u.nm).value >= wmin_with_margin.to(u.nm).value) & (
                    pl_waves.to(u.nm).value <= wmax_with_margin.to(u.nm).value)
        pl_waves_cropped = pl_waves[pl_mask]

        pixPerLens = inst["lenslSamp"]  # Number of pixels per spectral resolution elements
        delta_lambda = lambda_center / inst["Rs"]  # resolution element width
        pixel_spacing = delta_lambda / pixPerLens  # wavelength spacing per pixel
        num_pixels = int(np.floor((max_wave_bandpass - min_wave_bandpass) / pixel_spacing)) + 1
        data_waves = min_wave_bandpass + pixel_spacing * np.arange(num_pixels)
        data_waves_diff = np.diff(data_waves, prepend=2 * data_waves[0] - data_waves[1])

        ########
        ## This part broadens and scale the planet albedo spectral template including all the molecules ("pl0")
        pl0_template = pl_template[0]
        pl0_template_cropped = pl0_template[pl_mask]
        # todo: include stellar template multiplication since the RV features will be shifted
        pl0_template_incl_star = pl0_template_cropped  # *star_template_resampled
        pl0_template_resamp = broaden_and_resample(data_waves, pl_waves_cropped, pl0_template_incl_star, inst["Rs"],
                                                   n_jobs=n_jobs, broaden_pixel=broaden_pixel)
        # apply filter profile to spectra
        pl0_template_filt = pl0_template_resamp * bandpass_func(data_waves)
        # Normalize and scale to photons/sec
        pl0_template_norm_factor = np.sum(pl0_template_filt)
        ########
        ########
        ## If molecular templates are available, then process those:
        pl_mol_template_filt = {}
        for pl_mol_template, mol_name in zip(pl_template[1::],pl_template_name[1::]):
            pl_mol_template_cropped = pl_mol_template[pl_mask]
            pl_mol_template_incl_star = pl_mol_template_cropped#*star_template_resampled
            pl_mol_template_resamp= broaden_and_resample(data_waves, pl_waves_cropped, pl_mol_template_incl_star, inst["Rs"],n_jobs=n_jobs, broaden_pixel = broaden_pixel)
            # apply filter profile to spectra
            pl_mol_template_filt[mol_name] = pl_mol_template_resamp*bandpass_func(data_waves)
        ########

        for j in range(len(sInds)):

            _, _, C_sp,C_extra = self.Cp_Cb_Csp(TL, sInds[j], fZ[j], JEZ[j], dMag[j], WA[j], mode, TK=TK,returnExtra=True)
            _C_p0 = C_extra["C_p0"]
            _C_sr =  C_extra["C_sr"]#*0.41616687/0.00069059
            _C_z =  C_extra["C_z"]
            _C_ez =  C_extra["C_ez"]
            _C_dc =  C_extra["C_dc"]
            _C_bl = C_extra["C_bl"]
            _C_star = C_extra["C_star"]
            Npix = C_extra["Npix"]
            if "override_local_starlight_flux_ratio" in mode["syst"].keys():
                _C_sr = _C_star* mode["syst"]["override_local_starlight_flux_ratio"]
                C_sp = _C_sr * TL.PostProcessing.ppFact_char(WA) * self.stabilityFact

            # Obtain the renormalized spectral template using the new method
            # JB note: apparently the resolution is about ~500, for now hard coding R_star = 500
            R_star_template = 500
            star_template_obj = TL.get_spectral_template(sInds[j], mode)
            # star_template_flux = Observation(star_template, mode["bandpass"], force="taper").integrate()
            star_waves = star_template_obj.waveset # has units
            star_template = star_template_obj(star_waves)
            if star_template.unit != synphot.units.PHOTLAM:
                # Just making sure that the spectrum is in PHOTLAM. Not sure if actually needed? TBchecked
                raise Exception("Units of star_template should be synphot.units.PHOTLAM, it is {0} instead".format(star_template.unit))
            star_template = star_template.value
            # normalize the star_template to have max flux of unity
            star_template = star_template / np.nanmax(star_template)
            if inst["Rs"] > R_star_template/2.:
                warnings.warn("Instrument resolution is higher than 1/2 the stellar template resolution.")
            # crop star template
            star_mask = (star_waves.to(u.nm).value >= wmin_with_margin.to(u.nm).value) & (star_waves.to(u.nm).value <= wmax_with_margin.to(u.nm).value)
            star_waves_cropped = star_waves[star_mask]
            star_template_cropped = star_template[star_mask]

            star_template_resampled = np.interp(
                pl_waves_cropped.to(u.nm).value,
                star_waves_cropped.to(u.nm).value,
                star_template_cropped
            )
            star_template_resamp = broaden_and_resample(data_waves, pl_waves_cropped, star_template_resampled,
                                                        inst["Rs"], n_jobs=n_jobs, broaden_pixel=broaden_pixel)
            # apply filter profile to spectra
            star_template_filt = star_template_resamp*bandpass_func(data_waves)
            # Normalize
            star_template_norma = star_template_filt / np.sum(star_template_filt)
            # Scale to photons/sec
            star_template_scaled_C_sr = star_template_norma * _C_sr
            star_template_scaled_C_sp = star_template_norma * C_sp
            _C_star_spec = star_template_norma * _C_star

            # Scale planet template to actual phot/s
            pl0_template_scaled_C_p0 = pl0_template_filt/pl0_template_norm_factor * _C_p0

            # exposure time
            if self.texp_flag:
                with np.errstate(divide="ignore", invalid="ignore"):
                    texp = 1 / _C_p0 / 10  # Use 1/C_p0 as frame time for photon counting
            else:
                texp = inst["texp"].to(u.s)
            # readout noise
            _C_rn_spec = np.full_like(pl0_template_scaled_C_p0, Npix * inst["sread"] / texp)

            # clock-induced-charge
            _C_cc_spec = np.full_like(pl0_template_scaled_C_p0, Npix * inst["CIC"] / texp)

            # Dark current
            _C_dc_spec = np.full_like(pl0_template_scaled_C_p0, _C_dc)


            # zodi and exozodi spectra. Assuming flat spectra for now.
            _C_z_spec = np.full_like(pl0_template_scaled_C_p0, _C_z / np.size(data_waves))
            _C_ez_spec = np.full_like(pl0_template_scaled_C_p0, _C_ez / np.size(data_waves))

            # Background leakage spectrum
            _C_bl_spec = np.full_like(pl0_template_scaled_C_p0, _C_bl / np.size(data_waves))

            # C_p = PLANET SIGNAL RATE
            # photon counting efficiency
            PCeff = inst["PCeff"]
            # radiation dosage
            radDos = mode["radDos"]
            # photon-converted 1 frame (minimum 1 photon)
            # there may be zeros in the denominator. Suppress the resulting warning:
            with np.errstate(divide="ignore", invalid="ignore"):
                phConv = np.clip(((pl0_template_scaled_C_p0 + star_template_scaled_C_sr + _C_z_spec + _C_ez_spec) / Npix * texp), 1, None)
            # net charge transfer efficiency
            with np.errstate(invalid="ignore"):
                NCTE = 1.0 + (radDos / 4.0) * 0.51296 * (np.log10(phConv) + 0.0147233)
            # planet signal rate
            pl0_template_scaled_C_p = pl0_template_scaled_C_p0 * PCeff * NCTE
            # possibility of Npix=0 may lead C_p to be nan.  Change these to zero instead.
            pl0_template_scaled_C_p[np.isnan(pl0_template_scaled_C_p)] = 0

            # C_b = NOISE VARIANCE RATE
            # corrections for Ref star Differential Imaging e.g. dMag=3 and 20% time on ref
            # k_SZ for speckle and zodi light, and k_det for detector
            k_SZ = (
                1.0 + 1.0 / (10 ** (0.4 * self.ref_dMag) * self.ref_Time)
                if self.ref_Time > 0
                else 1.0
            )
            k_det = 1.0 + self.ref_Time
            # calculate Cb
            ENF2 = inst["ENF"] ** 2
            _C_b_spec = k_SZ * ENF2 * (star_template_scaled_C_sr + _C_z_spec + _C_ez_spec + _C_bl_spec) + k_det * (
                    ENF2 * (_C_dc_spec + _C_cc_spec) + _C_rn_spec
            )
            if "use_ADI" in mode.keys():
                if mode["use_ADI"]:
                    # double noise of everything is using ADI
                    _C_z_spec *=2
                    _C_ez_spec *=2
                    _C_bl_spec *=2
                    _C_dc_spec *=2
                    _C_cc_spec *=2
                    _C_rn_spec *=2

                    _C_b_spec *=2
                    star_template_scaled_C_sp *=2
                    star_template_scaled_C_sr *=2

            # for characterization, Cb must include the planet
            if not (mode["detectionMode"]):
                _C_b_spec = _C_b_spec + ENF2 * pl0_template_scaled_C_p0

            if returnExtra:
                ########
                ## If molecular templates are available, then process those:
                pl_mol_template_scaled_C_p0 = {}
                pl_mol_template_scaled_C_p = {}
                # spectral_envelop = np.nanmax(pl_template,axis=0)
                for pl_mol_template, mol_name in zip(pl_template[1::],pl_template_name[1::]):
                    # Normalize and scale to photons/sec, but use normalization from original planet spectrum
                    pl_mol_template_scaled_C_p0[mol_name] = pl_mol_template_filt[mol_name]/pl0_template_norm_factor * _C_p0
                    # planet signal rate
                    pl_mol_template_scaled_C_p[mol_name] = pl_mol_template_scaled_C_p0[mol_name] * PCeff * NCTE
                    # possibility of Npix=0 may lead C_p to be nan.  Change these to zero instead.
                    pl_mol_template_scaled_C_p[mol_name][np.isnan(pl_mol_template_scaled_C_p[mol_name])] = 0

                pl_mol_template_scaled_C_p_list.append(pl_mol_template_scaled_C_p)

            star_template_scaled_C_sr_list.append(star_template_scaled_C_sr << self.inv_s)
            _C_z_spec_list.append(_C_z_spec << self.inv_s)
            _C_ez_spec_list.append(_C_ez_spec << self.inv_s)
            _C_dc_spec_list.append(_C_dc_spec << self.inv_s)
            _C_cc_spec_list.append(_C_cc_spec << self.inv_s)
            _C_rn_spec_list.append(_C_rn_spec << self.inv_s)
            _C_star_spec_list.append(_C_star_spec << self.inv_s)
            pl0_template_scaled_C_p0_list.append(pl0_template_scaled_C_p0 << self.inv_s)
            _C_bl_spec_list.append(_C_bl_spec << self.inv_s)

            pl0_template_scaled_C_p_list.append(pl0_template_scaled_C_p << self.inv_s)
            _C_b_spec_list.append(_C_b_spec << self.inv_s)
            star_template_scaled_C_sp_list.append(star_template_scaled_C_sp << self.inv_s)

        if returnExtra:
            # organize components into an optional fourth result
            C_spec_extra = dict(
                C_sr_spec=star_template_scaled_C_sr_list, # starlight before post-processing
                C_z_spec=_C_z_spec_list,
                C_ez_spec=_C_ez_spec_list,
                C_dc_spec=_C_dc_spec_list,
                C_cc_spec=_C_cc_spec_list,
                C_rn_spec=_C_rn_spec_list,
                C_star_spec=_C_star_spec_list,
                C_p0_spec=pl0_template_scaled_C_p0_list,
                C_bl_spec=_C_bl_spec_list,
                C_p_mol_spec = pl_mol_template_scaled_C_p_list,
                Npix_per_bin=Npix,
                k_SZ=k_SZ,
                k_det=k_det,
                ENF2=ENF2,
                lambda_center=lambda_center,
            )
            return data_waves, pl0_template_scaled_C_p_list, _C_b_spec_list, star_template_scaled_C_sp_list, C_spec_extra
        else:
            return data_waves, pl0_template_scaled_C_p_list, _C_b_spec_list, star_template_scaled_C_sp_list

    def calc_snr(self, TL, sInds, fZ, JEZ, dMag, WA, mode, TK=None, pl_waves = None, pl_template = None, R_pl_template=None,pl_template_name=None,
                 figs=None,n_jobs=-1,broaden_pixel=True,output_filename=None,config_json_path=None):
        """Calculate SNR of target systems for given integration time for a specific observing
        mode (imaging or characterization), based on Nemati 2014 (SPIE).

        Args:
            TL (TargetList module):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            fZ (astropy Quantity array):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            JEZ (astropy Quantity array):
                Intensity of exo-zodiacal light in units of ph/s/m2/arcsec2
            dMag (float ndarray):
                Differences in magnitude between planets and their host star
            WA (astropy Quantity array):
                Working angles of the planets of interest in units of arcsec
            mode (dict):
                Selected observing mode
            TK (TimeKeeping object):
                Optional TimeKeeping object (default None), used to model detector
                degradation effects where applicable.
            pl_waves (~numpy.ndarray(float)):
                Wavelength array of planet spectral tempalte
            pl_template (List of ~numpy.ndarray(float)):
                List of planet albeda spectral template. If more than one, this can include molecular templates.
            R_pl_template:
                Spectral resolution of the pl_template spectrum as way to know what is the maximum spectral resolution for this calculation.
            pl_template_name (List of str)
                List of the names for the spectral templates in pl_template.
                e.g. ["all","H2O","O2"]
            figs (list of figure object):
                TBD
            n_jobs (int):
                Number of parallel jobs (-1 = all cores).
                Not parallelized if 0.
            broaden_pixel (Boolean):
                If True, subsequently broadens the spectrum the insturment resolution and then to the pixel width. Otherwise,
                only broaden to the instrumental resolution, and effectively assume that the pixel broadening is included in it.
                If samples_only is not None, having broaden_pixel=True is much slower.
            output_filename (str):
                Path to the output text file where the results will be written. Existing files
                with the same name will be overwritten.
            config_json_path (str):
                Path to the JSON configuration file used to generate the results. This will be appended
                at the end of the output file as a commented block for reference.

        Returns:
            intTime (astropy Quantity array):
                Integration times in units of day

        """
        intTime = (mode["intTime"]*u.h << u.s)

        inst = mode["inst"]
        syst = mode["syst"]
        if "spectro" in inst["name"].lower():
            if isinstance(pl_template, (np.ndarray)):
                pl_template = [pl_template]
            if isinstance(pl_template_name, str):
                pl_template_name = [pl_template_name]

            # Define all the output arrays for the SNR
            SNR_dict = {}
            SNR = np.full(shape=len(sInds), fill_value=np.nan)
            SNR_dict["stars"] = TL.Name
            SNR_dict["SNR_"+pl_template_name[0]+"_avg_per_bin"] = np.full(shape=len(sInds), fill_value=np.nan)
            for _pl_name in pl_template_name:
                SNR_dict["SNR_"+_pl_name+"_ignore_corr"] = np.full(shape=len(sInds), fill_value=np.nan)
                SNR_dict["SNR_"+_pl_name+"_uncorr_small_scale"] = np.full(shape=len(sInds), fill_value=np.nan)
                SNR_dict["SNR_"+_pl_name+"_corr_large_scale"] = np.full(shape=len(sInds), fill_value=np.nan)
                SNR_dict["SNR_"+_pl_name+"_corr_test"] = np.full(shape=len(sInds), fill_value=np.nan)
                SNR_dict["SNR_"+_pl_name+"_corr"] = np.full(shape=len(sInds), fill_value=np.nan)

            out = self.Cp_Cb_Csp_spec(TL, sInds, fZ, JEZ, dMag, WA, mode, TK=TK, returnExtra=True,
                                      pl_waves=pl_waves, pl_template=pl_template,
                                      R_pl_template=R_pl_template, pl_template_name=pl_template_name,
                                      n_jobs=n_jobs, broaden_pixel=broaden_pixel)
            data_waves = out[0] # Wavelength sampling of the "data", ie the spectra below
            pl0_template_scaled_C_p_list = out[1]   # List of planet spectra (including PCeff * NCTE)
            _C_b_spec_list = out[2]  # List of white noise stddev spectra (including k_SZ, ENF2, k_det)
            star_template_scaled_C_sp_list = out[3] # List of residual starlight spectra, ie correlated noise (_C_sr * post processing factor * stability factor)

            C_extra = out[4] # The outputs in there do not typically include the photon counting detector stuff
            pl0_template_scaled_C_p0_list = C_extra["C_p0_spec"] # List of planet spectra (NOT including PCeff * NCTE)
            star_template_scaled_C_sr_list = C_extra["C_sr_spec"] # List of starlight spectra (before post-processing)
            _C_z_spec_list = C_extra["C_z_spec"] # List of Zodi spectra
            _C_ez_spec_list = C_extra["C_ez_spec"] # List of exzodi spectra
            _C_dc_spec_list = C_extra["C_dc_spec"] # List of dark current spectra
            _C_bl_spec_list = C_extra["C_bl_spec"]
            _C_star_spec_list = C_extra["C_star_spec"]
            _C_rn_spec_list = C_extra["C_rn_spec"] # List of read noise spectra
            _C_cc_spec_list = C_extra["C_cc_spec"] # List of clock-induced charge spectra
            Npix = C_extra["Npix_per_bin"]
            k_SZ = C_extra["k_SZ"]
            k_det = C_extra["k_det"]
            ENF2 = C_extra["ENF2"]
            lambda_center = C_extra["lambda_center"] # Center wavelength of the bandpass

            SNR_dict["C_planet"] = np.nansum(np.array(intTime * pl0_template_scaled_C_p_list),axis=1)
            SNR_dict["C_star"] = np.nansum(np.array(intTime * _C_star_spec_list),axis=1)
            SNR_dict["C_local_starlight"] = np.nansum(np.array(intTime * star_template_scaled_C_sr_list),axis=1)
            SNR_dict["C_correlated_speckles"] = np.nansum(np.array(intTime * star_template_scaled_C_sp_list),axis=1)
            SNR_dict["C_zodi"] = np.nansum(np.array(intTime * _C_z_spec_list),axis=1)
            SNR_dict["C_exozodi"] = np.nansum(np.array(intTime * _C_ez_spec_list),axis=1)
            SNR_dict["C_background_leakage"] = np.nansum(np.array(intTime * _C_bl_spec_list),axis=1)
            SNR_dict["C_readnoise"] = np.nansum(np.array(intTime * _C_rn_spec_list),axis=1)
            SNR_dict["C_dark"] = np.nansum(np.array(intTime * _C_dc_spec_list),axis=1)
            SNR_dict["C_CIC"] = np.nansum(np.array(intTime * _C_cc_spec_list),axis=1)
            print(SNR_dict["C_readnoise"],SNR_dict["C_dark"],SNR_dict["C_CIC"],SNR_dict["C_readnoise"]+SNR_dict["C_dark"]+SNR_dict["C_CIC"])

            pl_mol_template_scaled_C_p_list = C_extra["C_p_mol_spec"]

            for j in range(len(sInds)):
                # Just grab every single spectra for all the lists above
                pl0_template_scaled_C_p = pl0_template_scaled_C_p_list[j]
                _C_b_spec = _C_b_spec_list[j]
                star_template_scaled_C_sp = star_template_scaled_C_sp_list[j]
                pl0_template_scaled_C_p0 = pl0_template_scaled_C_p0_list[j]
                star_template_scaled_C_sr = star_template_scaled_C_sr_list[j]
                _C_z_spec = _C_z_spec_list[j]
                _C_ez_spec = _C_ez_spec_list[j]
                _C_dc_spec = _C_dc_spec_list[j]
                _C_bl_spec = _C_bl_spec_list[j]
                _C_star_spec = _C_star_spec_list[j]
                _C_rn_spec = _C_rn_spec_list[j]
                _C_cc_spec = _C_cc_spec_list[j]
                pl_mol_template_scaled_C_p = pl_mol_template_scaled_C_p_list[j]


                # Define the "model" vector, ie the sigla
                m = intTime * pl0_template_scaled_C_p
                # Define the corresponding noise vector
                s = np.sqrt(intTime * _C_b_spec + (intTime * star_template_scaled_C_sp)**2)

                # Compute SNR with matched filter formula ignoring any correlations
                SNR_dict["SNR_"+pl_template_name[0]+"_ignore_corr"][j] = np.sqrt(np.nansum(m**2/s**2))
                # SNR per spectral bin
                SNR_dict["SNR_"+pl_template_name[0]+"_avg_per_bin"][j] = np.nanmean(m/s)

                if "chromaticity_dwave_nm" in syst.keys():
                    inv_cov0,cov_matrix0,corr_matrix0 = self.compute_cov_matrices(data_waves, WA[j], syst["chromaticity_dwave_nm"],
                                                                               intTime * star_template_scaled_C_sp,
                                                                               np.sqrt(intTime * _C_b_spec))
                    # Broadband SNR accounting for the covariance.
                    SNR_dict["SNR_"+pl_template_name[0] + "_corr"][j] = np.sqrt(np.linalg.multi_dot([m.T,inv_cov0,m]))

                    #####
                    ## The following is trying to decompose the spectrum into a small scale and large scale features to
                    ## see where the signal is
                    #####

                    # Compute correlation wavelength scale due to general PSF magnification
                    corr_scale = 1.22/(2*np.sqrt(np.log(2))) * data_waves ** 2 / (self.pupilDiam * WA[j].to(u.rad).value)
                    corr_scale = corr_scale.decompose().to(u.nm)

                    # Set the maximum value of the correlation scale to the chromaticity scale
                    corr_scale = np.clip(corr_scale, 0, (syst["chromaticity_dwave_nm"] * u.nm).to(corr_scale.unit))

                    # Only go through the separation of the small/large scale if the there is a non-zero correlation length
                    if not np.any(corr_scale.to(u.nm).value < np.max(np.diff(data_waves.to(u.nm).value))):
                        # Convert to resolution
                        corr_R = data_waves.to(u.nm).value/corr_scale.to(u.nm).value

                        # Compute large scale ("ls") spectrum, ie the CORRELATED part of the spectrum
                        m_ls = broaden(data_waves, m, corr_R, kernel="gaussian",n_jobs=n_jobs)
                        corr_starlight_ls = broaden(data_waves, intTime * star_template_scaled_C_sp, corr_R, kernel="gaussian",n_jobs=n_jobs)
                        # Compute small scale ("ss") spectrum, ie the UN-correlated part of the spectrum
                        m_ss = m-m_ls

                        # Define wavelength sampling of the small scale spectrum, ie Nyquist sampling of the correlation scale
                        # We need to down sample the large scale spectrum, otherwise the covariance matrix risks to be poorly conditioned
                        ls_resolution = np.nanmedian(corr_R)
                        ls_pixel_spacing = lambda_center / ls_resolution  / 2
                        ls_num_pixels = int(np.floor((data_waves[-1] - data_waves[0]-ls_pixel_spacing) / ls_pixel_spacing)) + 1
                        ls_waves = data_waves[0]+ls_pixel_spacing/2. + ls_pixel_spacing * np.arange(ls_num_pixels)
                        # Resample large scale spectrum
                        m_ls = downsample_spectrum(data_waves.to_value(u.nm), m_ls, ls_waves.to_value(u.nm))
                        corr_starlight_ls = downsample_spectrum(data_waves.to_value(u.nm), corr_starlight_ls, ls_waves.to_value(u.nm))

                        # SNR only including the features with a spectral scale SMALLER than the correlation length (ie, HIGH-pass filtered)
                        s_ss = np.sqrt(intTime * _C_b_spec)
                        SNR_dict["SNR_"+pl_template_name[0] + "_uncorr_small_scale"][j] = np.sqrt(np.nansum(m_ss ** 2 / s_ss ** 2))

                        var_ls_uncorr = intTime * downsample_spectrum(data_waves.to_value(u.nm), _C_b_spec, ls_waves.to_value(u.nm))
                        inv_cov,cov_matrix,corr_matrix = self.compute_cov_matrices(ls_waves, WA[j], syst["chromaticity_dwave_nm"],
                                                                                   corr_starlight_ls,np.sqrt(var_ls_uncorr))

                        # SNR only including the features with a spectral scale LARGER than the correlation length  (ie, LOW-pass filtered)
                        SNR_dict["SNR_"+pl_template_name[0] + "_corr_large_scale"][j] = np.sqrt(np.linalg.multi_dot([m_ls.T,inv_cov,m_ls]))
                        # Simply combine the small scale and large scale SNRs in quadrature for comparison
                        SNR_dict["SNR_"+pl_template_name[0] + "_corr_test"][j] = np.sqrt(SNR_dict["SNR_"+pl_template_name[0] + "_corr_large_scale"][j]**2+
                                                                             SNR_dict["SNR_"+pl_template_name[0] + "_uncorr_small_scale"][j]**2)
                ########
                ## If molecular templates are available, then process those:
                for pl_mol_template, mol_name in zip(pl_template[1::],pl_template_name[1::]):

                    m = intTime * pl_mol_template_scaled_C_p[mol_name]
                    SNR_dict["SNR_"+mol_name+"_ignore_corr"][j] = np.sqrt(np.nansum(m**2/s**2))
                    SNR_dict["SNR_"+mol_name + "_corr"][j] = np.sqrt(np.linalg.multi_dot([m.T,inv_cov0,m]))

                    # Only go through the separation of the small/large scale if the there is a non-zero correlation length
                    if not np.any(corr_scale.to(u.nm).value < np.max(np.diff(data_waves.to(u.nm).value))):
                        m_ls = broaden(data_waves, m, corr_R, kernel="gaussian",n_jobs=n_jobs)
                        m_ss = m-m_ls
                        SNR_dict["SNR_"+mol_name + "_uncorr_small_scale"][j] = np.sqrt(np.nansum(m_ss ** 2 / s_ss ** 2))
                        m_ls = downsample_spectrum(data_waves.to_value(u.nm), m_ls, ls_waves.to_value(u.nm))
                        # print(data_waves.to_value(u.nm).shape,data_waves.to_value(u.nm))
                        # print(ls_waves.to_value(u.nm).shape,ls_waves.to_value(u.nm))
                        SNR_dict["SNR_"+mol_name + "_corr_large_scale"][j] = np.sqrt(np.linalg.multi_dot([m_ls.T,inv_cov,m_ls]))
                        SNR_dict["SNR_"+mol_name + "_corr_test"][j] = np.sqrt(SNR_dict["SNR_"+mol_name + "_corr_large_scale"][j]**2+
                                                                  SNR_dict["SNR_"+mol_name + "_uncorr_small_scale"][j]**2)

                # This is for the default SNR being returned by the function.
                # Use the SNR with the covariance is valid, otherwise the "ignore correlation" one.
                if np.isfinite(SNR_dict["SNR_"+pl_template_name[0]+"_corr"][j]):
                    SNR[j] = SNR_dict["SNR_" + pl_template_name[0] + "_corr"][j]
                else:
                    SNR[j] = SNR_dict["SNR_"+pl_template_name[0]+"_ignore_corr"][j]

                if figs is not None:

                    import matplotlib.pyplot as plt
                    plt.figure(figs[j])
                    # plt.subplot(3,1,1)
                    # plt.title(_pl_name+" spectral templates")
                    # plt.plot(bandpass_waves.to(u.nm).value, bandpass_filter.value, label="bandpass filter")
                    # plt.plot(star_waves_cropped.to(u.nm).value, star_template_cropped, label="Original stellar template")
                    # plt.plot(pl_waves_cropped.to(u.nm).value, pl0_template_cropped, label="Original planet template")
                    # # plt.plot(pl_waves_cropped.to(u.nm).value, pl_template_incl_star, label="pl_template_incl_star")
                    # # # plt.plot(pl_waves_cropped.to(u.nm).value, pl_template_broadR, label="pl_template_broadR")
                    # # # plt.plot(pl_waves_cropped.to(u.nm).value, pl_template_broadpix, label="pl_template_broadpix")
                    # # plt.plot(data_waves.to(u.nm).value, pl_template_resamp, label="pl_template_resamp")
                    # # plt.plot(data_waves.to(u.nm).value, star_template_resamp, label="star_template_resamp")
                    # plt.plot(data_waves.to(u.nm).value, pl0_template_filt, label="Observed planet template")
                    # for pl_mol_template, mol_name in zip(pl_template[1::],pl_template_name[1::]):
                    #     plt.plot(data_waves.to(u.nm).value, pl_mol_template_filt[mol_name], label=mol_name)
                    # plt.plot(data_waves.to(u.nm).value, star_template_filt, label="Observed star template")
                    # # plt.scatter(mode["bandpass"].avgwave().to(u.nm).value, weighted_albedo.value,
                    # #             label="planet phot bandpass")
                    # plt.xlabel(f"Wavelength [nm]")
                    # plt.ylabel("Arbitrary flux")
                    # plt.legend()
                    # plt.grid(True)

                    # plt.subplot(3,1,2)
                    # plt.title("Planet and molecules")
                    # # _C_b_spec = k_SZ * ENF2 * (star_template_scaled_C_sr + _C_z_spec + _C_ez_spec + _C_bl_spec) + k_det * (
                    # #         ENF2 * (_C_dc_spec + _C_cc_spec) + _C_rn_spec
                    # plt.plot(data_waves.to(u.nm).value, intTime * k_SZ * ENF2 * pl0_template_scaled_C_p0,"o", label="Planet")
                    # for pl_mol_template, mol_name in zip(pl_template[1::],pl_template_name[1::]):
                    #     plt.plot(data_waves.to(u.nm).value, intTime * k_SZ * ENF2 * pl_mol_template_scaled_C_p0[mol_name], label=mol_name)
                    # plt.legend()

                    # plt.subplot(3,1,3)
                    plt.title("Simulation; Exposure time: {0}; Spectral resolution: {1:.0f}".format(intTime,inst["Rs"]))
                    # _C_b_spec = k_SZ * ENF2 * (star_template_scaled_C_sr + _C_z_spec + _C_ez_spec + _C_bl_spec) + k_det * (
                    #         ENF2 * (_C_dc_spec + _C_cc_spec) + _C_rn_spec
                    plt.plot(data_waves.to(u.nm).value, intTime * pl0_template_scaled_C_p,"o", label="Planet",color="blue")
                    plt.plot(data_waves.to(u.nm).value, intTime * star_template_scaled_C_sr,"*", label="Starlight (before subtraction)",color="red")

                    plt.plot(data_waves.to(u.nm).value, np.sqrt(intTime * ENF2 * pl0_template_scaled_C_p0),"--", label="Planet (stddev)")
                    plt.plot(data_waves.to(u.nm).value, np.sqrt(intTime * k_SZ * ENF2 * star_template_scaled_C_sr),"--", label="Starlight (stddev)")
                    plt.plot(data_waves.to(u.nm).value, np.sqrt(intTime * k_det * ENF2 * _C_dc_spec),"--", label="Dark current (stddev)")
                    plt.plot(data_waves.to(u.nm).value, np.sqrt(intTime * k_det * ENF2 * _C_cc_spec),"--", label="clock-induced-charge (stddev)")
                    plt.plot(data_waves.to(u.nm).value, np.sqrt(intTime * k_det * _C_rn_spec),"--", label="Read noise (stddev)")
                    plt.plot(data_waves.to(u.nm).value, np.sqrt(intTime * k_SZ * ENF2 * _C_z_spec),"--", label="Zodi (stddev)")
                    plt.plot(data_waves.to(u.nm).value, np.sqrt(intTime * k_SZ * ENF2 * _C_ez_spec),"--", label="Exozodi (stddev)")
                    plt.plot(data_waves.to(u.nm).value, np.sqrt(intTime * k_SZ * ENF2 * _C_bl_spec),"--", label="Background leakage (stddev)")

                    plt.plot(data_waves.to(u.nm).value, np.sqrt(intTime * _C_b_spec),"x", label="Total uncorrelated noise (stddev)",color="black")
                    plt.plot(data_waves.to(u.nm).value, intTime * star_template_scaled_C_sp,".", label="Correlated noise (stddev)",color="orange") # Why is not multiplied by k_SZ * ENF2?
                    plt.xlabel(f"Wavelength [nm]")
                    plt.ylabel(f"Flux [Photons]")
                    plt.yscale("log")
                    plt.legend()
                    plt.grid(True)
                    # plt.tight_layout()
                    # plt.show()
            if output_filename is not None:
                write_snr_results_to_file(SNR_dict, output_filename, config_json_path=config_json_path)
            return SNR
        else:
            # electron counts
            C_p, C_b, C_sp = self.Cp_Cb_Csp(TL, sInds, fZ, JEZ, dMag, WA, mode, TK=TK)
            _C_p = C_p.to_value(self.inv_s)
            _C_b = C_b.to_value(self.inv_s)
            _C_sp = C_sp.to_value(self.inv_s)

            # get SNR threshold
            intTime_sec = (mode["intTime"]*u.h << u.s).value
            # calculate integration time based on Nemati 2014
            with np.errstate(divide="ignore", invalid="ignore"):
                if mode["syst"]["occulter"] is False:
                    SNR = np.true_divide(_C_p*intTime_sec,np.sqrt(_C_b*intTime_sec+(_C_sp*intTime_sec)**2))
                else:
                    SNR = np.true_divide(_C_p*intTime_sec,np.sqrt(_C_b*intTime_sec))


            return SNR

    def compute_cov_matrices(self,ls_waves, WA, chromaticity_dwave_nm, std_corr, std_uncorr):
        """
        todo Write documentation
        :param mode:
        :param ls_waves:
        :param WA:
        :param std_ls_corr:
        :param var_ls_uncorr:
        :return:
        """

        mean_ls_wave_matrix = (ls_waves[:, None] + ls_waves[None, :]) / 2
        corr_scale_matrix = np.sqrt(2) * mean_ls_wave_matrix ** 2 / (self.pupilDiam * WA.to(u.rad).value)
        corr_scale_matrix = corr_scale_matrix.decompose().to(u.nm)
        # Set the maximum value of the correlation scale to the chromaticity scale
        corr_scale_matrix = np.clip(corr_scale_matrix, 0,
                                    (chromaticity_dwave_nm* u.nm).to(corr_scale_matrix.unit))
        diff_ls_wave_matrix = np.abs(ls_waves[:, None] - ls_waves[None, :])
        corr_matrix = np.exp(-0.5 * diff_ls_wave_matrix.to_value(u.nm) ** 2 / corr_scale_matrix.to_value(u.nm) ** 2)
        # where there was a 0/0 in the diagonal, set to unity:
        corr_matrix[np.where((diff_ls_wave_matrix.to_value(u.nm)==0) * (corr_scale_matrix.to_value(u.nm) ==0))] = 1

        cov_matrix = np.diag(std_uncorr**2) + (std_corr[:, None] * std_corr[None, :]) * corr_matrix
        cov_matrix = cov_matrix.value

        # (sign, logdet) = np.linalg.slogdet(cov_matrix)
        # print(sign, logdet)
        cond_number = np.linalg.cond(cov_matrix, p=2)
        if cond_number > 1e6:
            # regularize covariance and inverse
            inv_cov, regularized_cov, eigvecs = regularized_inverse(cov_matrix, threshold=1e-6)
        else:
            inv_cov = np.linalg.inv(cov_matrix)

        return inv_cov,cov_matrix,corr_matrix

    def calc_intTime(self, TL, sInds, fZ, JEZ, dMag, WA, mode, TK=None):
        """Finds integration times of target systems for a specific observing
        mode (imaging or characterization), based on Nemati 2014 (SPIE).

        Args:
            TL (TargetList module):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            fZ (astropy Quantity array):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            JEZ (astropy Quantity array):
                Intensity of exo-zodiacal light in units of ph/s/m2/arcsec2
            dMag (float ndarray):
                Differences in magnitude between planets and their host star
            WA (astropy Quantity array):
                Working angles of the planets of interest in units of arcsec
            mode (dict):
                Selected observing mode
            TK (TimeKeeping object):
                Optional TimeKeeping object (default None), used to model detector
                degradation effects where applicable.

        Returns:
            intTime (astropy Quantity array):
                Integration times in units of day

        """
        # raise Exception("Not implemented yet")

        # electron counts
        C_p, C_b, C_sp = self.Cp_Cb_Csp(TL, sInds, fZ, JEZ, dMag, WA, mode, TK=TK)
        _C_p = C_p.to_value(self.inv_s)
        _C_b = C_b.to_value(self.inv_s)
        _C_sp = C_sp.to_value(self.inv_s)

        # get SNR threshold
        SNR = mode["SNR"]
        # calculate integration time based on Nemati 2014
        with np.errstate(divide="ignore", invalid="ignore"):
            if mode["syst"]["occulter"] is False:
                intTime = (
                    np.true_divide(SNR**2.0 * _C_b, (_C_p**2.0 - (SNR * _C_sp) ** 2.0))
                    * self.s2d
                )
            else:
                intTime = np.true_divide(SNR**2.0 * _C_b, (_C_p**2.0)) * self.s2d
        # infinite and NAN are set to zero
        intTime[np.isinf(intTime) | np.isnan(intTime)] = np.nan
        # negative values are set to zero
        intTime[intTime < 0.0] = np.nan

        return intTime << u.d

    def calc_dMag_per_intTime(
        self,
        intTimes,
        TL,
        sInds,
        fZ,
        JEZ,
        WA,
        mode,
        C_b=None,
        C_sp=None,
        TK=None,
        analytic_only=False,
    ):
        """Finds achievable dMag for one integration time per star in the input
        list at one working angle.

        Args:
            intTimes (astropy Quantity array):
                Integration times
            TL (TargetList module):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            fZ (astropy Quantity array):
                Surface brightness of local zodiacal light for each star in sInds
                in units of 1/arcsec2
            JEZ (astropy Quantity array):
                Intensity of exo-zodiacal light in units of ph/s/m2/arcsec2
            WA (astropy Quantity array):
                Working angle for each star in sInds in units of arcsec
            mode (dict):
                Selected observing mode
            C_b (astropy Quantity array):
                Background noise electron count rate in units of 1/s (optional)
            C_sp (astropy Quantity array):
                Residual speckle spatial structure (systematic error) in units of 1/s
                (optional)
            TK (TimeKeeping object):
                Optional TimeKeeping object (default None), used to model detector
                degradation effects where applicable.
            analytic_only (Bool):
                Returns the analytic solution without running root-finding

        Returns:
            dMag (ndarray):
                Achievable dMag for given integration time and working angle

        """
        # Calculate the analytic values for the dMag and the singularity
        # dMag value (which functions as a bound)
        dMag_x0s, sing_x0s = self.dMag_per_intTime_x0(
            TL, sInds, fZ, JEZ, WA, mode, TK, intTimes
        )
        if analytic_only:
            return dMag_x0s
        # Loop over every star given and numerically refine the dMag
        dMags = np.zeros(len(sInds))
        for i, int_time in enumerate(tqdm(intTimes, delay=2)):
            if int_time == 0:
                warnings.warn(
                    "calc_dMag_per_int_time got an intTime=0 input, nan returned"
                )
                dMags[i] = np.nan
                continue
            if (WA[i] > mode["OWA"]) or (WA[i] < mode["IWA"]):
                warnings.warn(
                    "calc_dMag_per_int_time got WA not in [IWA, OWA], nan returned"
                )
                dMags[i] = np.nan
                continue
            # Parameters for this star
            s = [sInds[i]]

            args_denom = (TL, s, fZ[i].ravel(), JEZ[i].ravel(), WA[i].ravel(), mode, TK)
            args_intTime = (*args_denom, int_time.ravel())

            # Refine the singularity dMag value with root finding
            if mode["syst"]["occulter"]:
                singularity_dMag = np.inf
            else:
                try:
                    singularity_res = root_scalar(
                        self.int_time_denom_obj,
                        x0=sing_x0s[i],
                        args=args_denom,
                        bracket=[0, 50],
                    )
                except ValueError:
                    singularity_dMag = np.inf
                    dMags[i] = np.nan
                    continue
                else:
                    singularity_dMag = singularity_res.root

            if int_time == np.inf:
                dMag = singularity_dMag
            else:
                # Adjust the lower bounds until we have proper convergence
                star_vmag = TL.Vmag[sInds[i]]
                test_lb_subtractions = [2, 10]
                converged = False
                for j, lb_subtraction in enumerate(test_lb_subtractions):
                    initial_lower_bound = np.clip(
                        singularity_dMag - lb_subtraction - star_vmag,
                        5,  # Lower bound (of the lower bound) of 5
                        dMag_x0s[i] - 2,  # Upper bound of 2 under the analytic value
                    )
                    lb_adjustment = 0
                    while not converged:
                        dMag_lb = initial_lower_bound + lb_adjustment

                        if dMag_lb > singularity_dMag:
                            if j == len(test_lb_subtractions) - 1:
                                raise ValueError(
                                    (
                                        "No dMag convergence for"
                                        f" {mode['instName']}, sInds {sInds[i]}, "
                                        f"int_times {int_time}, and WA {WA[i]}"
                                    )
                                )
                            else:
                                break
                        dMag_min_res = minimize(
                            self.dMag_per_intTime_obj,
                            dMag_x0s[i],
                            args=args_intTime,
                            bounds=[(dMag_lb, singularity_dMag)],
                            method="L-BFGS-B",
                            tol=1e-10,
                        )

                        # Some times minimize_scalar returns the x value in an
                        # array and sometimes it doesn't, idk why
                        if isinstance(dMag_min_res["x"], np.ndarray):
                            dMag = dMag_min_res["x"][0]
                        else:
                            dMag = dMag_min_res["x"]

                        # Check if the returned time difference is greater than 5%
                        # of the true int time, if it is then raise the lower bound
                        # and try again. Also, if it converges to the lower bound
                        # then raise the lower bound and try again
                        time_diff = dMag_min_res["fun"]
                        if (time_diff > int_time.to(u.day).value / 20) or (
                            np.abs(dMag - dMag_lb) < 0.01
                        ):
                            lb_adjustment += 1
                        else:
                            converged = True
            dMags[i] = dMag

        return dMags

    def ddMag_dt(
        self, intTimes, TL, sInds, fZ, JEZ, WA, mode, C_b=None, C_sp=None, TK=None
    ):
        """Finds derivative of achievable dMag with respect to integration time

        Args:
            intTimes (astropy Quantity array):
                Integration times
            TL (TargetList module):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            fZ (astropy Quantity array):
                Surface brightness of local zodiacal light for each star in sInds
                in units of 1/arcsec2
            JEZ (astropy Quantity array):
                Intensity of exo-zodiacal light in units of ph/s/m2/arcsec2
            WA (astropy Quantity array):
                Working angle for each star in sInds in units of arcsec
            mode (dict):
                Selected observing mode
            C_b (astropy Quantity array):
                Background noise electron count rate in units of 1/s (optional)
            C_sp (astropy Quantity array):
                Residual speckle spatial structure (systematic error) in units of 1/s
                (optional)
            TK (TimeKeeping object):
                Optional TimeKeeping object (default None), used to model detector
                degradation effects where applicable.

        Returns:
            ddMagdt (ndarray):
                Derivative of achievable dMag with respect to integration time

        """

        # cast sInds, WA, fZ, fEZ, and intTimes to arrays
        sInds = np.array(sInds, ndmin=1, copy=copy_if_needed)
        WA = np.array(WA.value, ndmin=1) * WA.unit
        fZ = np.array(fZ.value, ndmin=1) * fZ.unit
        JEZ = np.array(JEZ.value, ndmin=1) * JEZ.unit
        intTimes = np.array(intTimes.value, ndmin=1) * intTimes.unit
        assert len(intTimes) == len(sInds), "intTimes and sInds must be same length"
        assert len(JEZ) == len(sInds), "JEZ must be an array of length len(sInds)"
        assert len(fZ) == len(sInds), "fZ must be an array of length len(sInds)"
        assert len(WA) == len(sInds), "WA must be an array of length len(sInds)"

        rough_dMag = np.zeros(len(sInds)) + 25.0
        if (C_b is None) or (C_sp is None):
            _, C_b, C_sp = self.Cp_Cb_Csp(
                TL, sInds, fZ, JEZ, rough_dMag, WA, mode, TK=TK
            )
        ddMagdt = (
            2.5
            / (2.0 * np.log(10.0))
            * (C_b / (C_b * intTimes + (C_sp * intTimes) ** 2.0)).to_value(self.inv_s)
        )

        return ddMagdt / u.s

    def dMag_per_intTime_obj(self, dMag, *args):
        """
        Objective function for calc_dMag_per_intTime's minimize_scalar function
        that uses calc_intTime from Nemati and then compares the value to the
        true intTime value

        Args:
            dMag (~numpy.ndarray(float)):
                dMag being tested
            *args:
                all the other arguments that calc_intTime needs

        Returns:
            ~numpy.ndarray(float):
                Absolute difference between true and evaluated integration time in days.
        """
        TL, sInds, fZ, JEZ, WA, mode, TK, true_intTime = args
        est_intTime = self.calc_intTime(TL, sInds, fZ, JEZ, dMag, WA, mode, TK)
        abs_diff = np.abs(true_intTime.to_value(u.day) - est_intTime.to_value(u.day))
        return abs_diff

    def calc_saturation_dMag(self, TL, sInds, fZ, JEZ, WA, mode, TK=None):
        """
        This calculates the delta magnitude for each target star that
        corresponds to an infinite integration time.

        Args:
            TL (:ref:`TargetList`):
                TargetList class object
            sInds (numpy.ndarray(int)):
                Integer indices of the stars of interest
            fZ (~astropy.units.Quantity(~numpy.ndarray(float))):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            JEZ (astropy Quantity array):
                Intensity of exo-zodiacal light in units of ph/s/m2/arcsec2
            WA (~astropy.units.Quantity(~numpy.ndarray(float))):
                Working angles of the planets of interest in units of arcsec
            mode (dict):
                Selected observing mode
            TK (:ref:`TimeKeeping`, optional):
                Optional TimeKeeping object (default None), used to model detector
                degradation effects where applicable.

        Returns:
            ~numpy.ndarray(float):
                Maximum achievable dMag for  each target star
        """

        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=copy_if_needed)

        # TODO: revisit this if updating occulter noise floor model
        if mode["syst"].get("occulter"):
            saturation_dMag = np.full(shape=len(sInds), fill_value=np.inf)
        else:
            saturation_dMag = np.zeros(len(sInds))
            for i, sInd in enumerate(tqdm(sInds, desc="Calculating saturation_dMag")):
                args = (
                    TL,
                    [sInd],
                    [fZ[i].value] * fZ.unit,
                    [JEZ[i].value] * JEZ.unit,
                    [WA[i].value] * WA.unit,
                    mode,
                    TK,
                )
                singularity_res = root_scalar(
                    self.int_time_denom_obj,
                    args=args,
                    method="brentq",
                    bracket=[10, 40],
                )
                singularity_dMag = singularity_res.root
                saturation_dMag[i] = singularity_dMag

        return saturation_dMag

    def int_time_denom_obj(self, dMag, *args):
        """
        Objective function for calc_dMag_per_intTime's calculation of the root
        of the denominator of calc_inTime to determine the upper bound to use
        for minimizing to find the correct dMag. Only necessary for coronagraphs.

        Args:
            dMag (~numpy.ndarray(float)):
                dMag being tested
            *args:
                all the other arguments that calc_intTime needs

        Returns:
            ~astropy.units.Quantity(~numpy.ndarray(float)):
                Denominator of integration time expression
        """
        TL, sInds, fZ, JEZ, WA, mode, TK = args
        C_p, C_b, C_sp = self.Cp_Cb_Csp(TL, sInds, fZ, JEZ, dMag, WA, mode, TK=TK)
        denom = (
            C_p.to_value(self.inv_s) ** 2
            - (mode["SNR"] * C_sp.to_value(self.inv_s)) ** 2
        )
        return denom

    def dMag_per_intTime_x0(self, TL, sInds, fZ, JEZ, WA, mode, TK, intTime):
        """
        This calculates the initial guess for the dMag for each target star
        that corresponds to an infinite integration time.
        Args:
            TL (:ref:`TargetList`):
                TargetList class object
            sInds (numpy.ndarray(int)):
                Integer indices of the stars of interest
            fZ (~astropy.units.Quantity(~numpy.ndarray(float))):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            JEZ (astropy Quantity array):
                Intensity of exo-zodiacal light in units of ph/s/m2/arcsec2
            WA (~astropy.units.Quantity(~numpy.ndarray(float))):
                Working angles of the planets of interest in units of arcsec
            mode (dict):
                Selected observing mode
            TK (:ref:`TimeKeeping`, optional):
                Optional TimeKeeping object (default None), used to model detector
                degradation effects where applicable.
            intTime (astropy Quantity array):
                Integration time
        Returns:
            tuple:
                ~numpy.ndarray(float):
                    Initial guess for dMag for each target star
                ~numpy.ndarray(float):
                    Initial guess for dMag for each target star that corresponds
                    to an infinite integration time
        """
        inst = mode["inst"]
        syst = mode["syst"]
        lam = mode["lam"]
        tmp_dMags = np.full(len(sInds), 25)
        Cp, Cb, Csp, C_extra = self.Cp_Cb_Csp(
            TL, sInds, fZ, JEZ, tmp_dMags, WA, mode, returnExtra=True, TK=TK
        )
        _Csp = Csp.to_value(self.inv_s)

        k_SZ = (
            1.0 + 1.0 / (10 ** (0.4 * self.ref_dMag) * self.ref_Time)
            if self.ref_Time > 0
            else 1.0
        )
        k_det = 1.0 + self.ref_Time
        ENF2 = inst["ENF"] ** 2
        SNR = mode["SNR"]

        _Cstar = C_extra["C_star"].to_value(self.inv_s)
        _Csr = C_extra["C_sr"].to_value(self.inv_s)
        _Cz = C_extra["C_z"].to_value(self.inv_s)
        _Cez = C_extra["C_ez"].to_value(self.inv_s)
        _Cbl = C_extra["C_bl"].to_value(self.inv_s)
        _Cdc = C_extra["C_dc"].to_value(self.inv_s)
        _Ccc = C_extra["C_cc"].to_value(self.inv_s)
        _Crn = C_extra["C_rn"].to_value(self.inv_s)
        Npix = C_extra["Npix"]
        _intTime = intTime.to_value(u.s)

        a0 = 0
        a1 = k_SZ * ENF2 * (_Csr + _Cz + _Cez + _Cbl) + k_det * ENF2 * _Cdc
        if self.texp_flag:
            # When using texp_flag the frame time is 1/(10*C_p0) which affects
            # the clock induced charge and the read noise terms
            a0 += 10 * k_det * Npix * (ENF2 * inst["CIC"] + inst["sread"])
        else:
            a1 += k_det * (ENF2 * _Ccc + _Crn)
        if not mode["detectionMode"]:
            # Account for the direct addition of the planet signal to the
            # background noise
            a0 += ENF2

        # Calculating terms necessary for the inversion
        radDos = mode["radDos"]
        _texp = inst["texp"].to_value(u.s)
        with np.errstate(divide="ignore", invalid="ignore"):
            # Making an approximation for the planet signal at 23 dMag
            approx_Cp0 = _Cstar * 10 ** (-0.4 * 23) * syst["core_thruput"](lam, WA)
            phConv = np.clip(
                ((approx_Cp0 + _Csr + _Cz + _Cez) / Npix * _texp),
                1,
                None,
            )
            # Form of phConv that includes a fainter planet signal to use for
            # the singularity calculation
            approx_Cp_inf = _Cstar * 10 ** (-0.4 * 30) * syst["core_thruput"](lam, WA)
            phConv_inf = np.clip(
                ((approx_Cp_inf + _Csr + _Cz + _Cez) / Npix * _texp),
                1,
                None,
            )
        # net charge transfer efficiency
        with np.errstate(invalid="ignore"):
            NCTE = 1.0 + (radDos / 4.0) * 0.51296 * (np.log10(phConv) + 0.0147233)
            NCTE_inf = 1.0 + (radDos / 4.0) * 0.51296 * (
                np.log10(phConv_inf) + 0.0147233
            )
        A = (inst["PCeff"] * NCTE * _intTime / SNR) ** 2
        B = -a0 * _intTime
        C = -a1 * _intTime
        if not mode["syst"]["occulter"]:
            # Account for speckle noise
            C -= (_Csp * _intTime) ** 2
        # First root (the positive one)
        C_p01 = (-B + np.sqrt(B**2 - 4 * A * C)) / (2 * A)

        core_thruput = syst["core_thruput"](lam, WA)
        dMag0 = -2.5 * np.log10(C_p01 / (_Cstar * core_thruput))

        # Calculate the dMag corresponding to the singularity (infinite intTime)
        C_p0inf = SNR * _Csp / (inst["PCeff"] * NCTE_inf)
        if not mode["syst"]["occulter"]:
            dMag_inf = -2.5 * np.log10(C_p0inf / (_Cstar * core_thruput))
        else:
            # The occulter has no speckle noise in the current formulation
            dMag_inf = np.full(len(sInds), np.inf)

        return dMag0, dMag_inf



def broaden(
    wavelengths, spectrum, resolution_array, window=5, n_jobs=-1, kernel="gaussian",wave_samples_only = None,
):
    """
    NaN-resistant, parallel spectral broadening on non-uniform grid with resolution input.
    Supports both Gaussian and boxcar (top-hat) broadening. Edges are padded with constant extension.

    Parameters
    ----------
    wavelengths : ndarray
        Wavelength array (1D, non-uniform allowed).
    spectrum : ndarray
        Flux values (1D), may contain NaNs.
    resolution_array : ndarray
        Resolving power R = lam / dlam at each wavelength.
    window : float
        For Gaussian: width of convolution window in sigma (default 5).
        For box: total width is box_width = lambda / R, window is ignored.
    n_jobs : int
        Number of parallel jobs (-1 = all cores).
    kernel : str
        'gaussian' or 'box'
    wave_samples_only: ndarray
        Only compute the broadened spectrum for the wavelengths values of "wavelengths" bracketing the values in
        wave_samples_only. The output arrays are the same size either way, but if wave_samples_only is not None, most
        values are likely to be nans.

    Returns
    -------
    broadened_spectrum : ndarray
        Broadened 1D spectrum.
    """
    if isinstance(spectrum, Quantity):
        spectrum_nounit = spectrum.value
        spectrum_unit = spectrum.unit
    else:
        spectrum_nounit = spectrum
        spectrum_unit = 1

    N = len(wavelengths)
    assert N == len(spectrum) == len(resolution_array)
    assert kernel in {"gaussian", "box"}

    # Estimate max width for padding
    width_max = np.max(wavelengths / resolution_array)
    sigma_max = width_max / 2.3548 if kernel == "gaussian" else width_max / 2
    delta_pad = window * sigma_max if kernel == "gaussian" else sigma_max

    # Pad spectrum and wavelengths
    lam_start, lam_end = wavelengths[0], wavelengths[-1]
    dlam = np.median(np.diff(wavelengths))
    n_pad = int(np.ceil(delta_pad / dlam))

    left_pad_wl = lam_start - dlam * np.arange(n_pad, 0, -1)
    right_pad_wl = lam_end + dlam * np.arange(1, n_pad + 1)

    wl_pad = np.concatenate([left_pad_wl, wavelengths, right_pad_wl])
    sp_pad = np.concatenate([
        np.full(n_pad, spectrum_nounit[np.where(np.isfinite(spectrum_nounit))[0][0]]),
        spectrum_nounit,
        np.full(n_pad, spectrum_nounit[np.where(np.isfinite(spectrum_nounit))[0][-1]])
    ])
    wl_pad_diff = np.diff(wl_pad, prepend=2 * wl_pad[0] - wl_pad[1])

    if wave_samples_only is not None:
        indices_to_process = find_bracketing_indices(wavelengths.to_value(u.nm), wave_samples_only.to_value(u.nm))
    else:
        indices_to_process = np.arange(N)

    def broaden_at_index(i):
        lam_i = wavelengths[i]
        R_i = resolution_array[i]

        if kernel == "gaussian":
            fwhm_i = lam_i / R_i
            sigma_i = fwhm_i / 2.3548
            delta = window * sigma_i
            mask = np.abs(wl_pad - lam_i) < delta
            wl_local = wl_pad[mask]
            wl_local_diff = wl_pad_diff[mask]
            sp_local = sp_pad[mask]
            valid = np.isfinite(sp_local)
            if not np.any(valid):
                return np.nan
            weights = (
                np.exp(-0.5 * ((wl_local - lam_i) / sigma_i) ** 2)
                / np.sqrt(2 * np.pi * sigma_i**2)
                * wl_local_diff
            )
            weights = weights[valid]
            return np.sum(weights * sp_local[valid])

        elif kernel == "box":
            width = lam_i / R_i
            half_width = width / 2
            mask = np.abs(wl_pad - lam_i) < half_width
            wl_local = wl_pad[mask]
            wl_local_diff = wl_pad_diff[mask]
            sp_local = sp_pad[mask]
            valid = np.isfinite(sp_local)
            if not np.any(valid):
                return np.nan
            # weights = wl_local_diff[valid]/(width)
            weights = wl_local_diff[valid]/np.sum(wl_local_diff)
            # print((width,np.sum(wl_local_diff)))
            return np.sum(weights * sp_local[valid])

    # Assume indices_to_process is a list or array of valid indices
    broadened = np.full(N, np.nan)  # pre-fill with NaNs

    if n_jobs == 0:
        results = np.full(np.size(indices_to_process),np.nan)
        for k,i in enumerate(indices_to_process):
            results[k] = broaden_at_index(i)
    else:
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(broaden_at_index)(i) for i in indices_to_process
        )

    # Fill the processed results into the output array
    broadened[indices_to_process] = results

    return np.array(broadened)*spectrum_unit


def broaden_and_resample(data_waves, waves, spectrum, R,n_jobs=-1, broaden_pixel = False):
    """
    Broaden an input spectrum to the instrumental resolution and pixel resolution, then resample.

    Parameters
    ----------
    data_waves : Quantity
        Target wavelength array to which the final spectrum will be resampled (with units).
    waves : Quantity
        Original wavelength array of the input spectrum (with units).
    spectrum : ndarray
        Input spectrum values sampled at `waves`.
    R : float
        Instrumental spectral resolution R = lambda / d_lambda.
    n_jobs : int
        Number of parallel jobs (-1 = all cores).
    broaden_pixel: Bool
        If True, subsequently broadens the spectrum the insturment resolution and then to the pixel width. Otherwise,
        only broaden to the instrumental resolution, and effectively assume that the pixel broadening is included in it.
        If samples_only is not None, having broaden_pixel=True is much slower.

    Returns
    -------
    spectrum_resamp : ndarray
        Spectrum broadened and resampled at `data_waves`.
    """

    if isinstance(spectrum, Quantity):
        spectrum_nounit = spectrum.value
        spectrum_unit = spectrum.unit
    else:
        spectrum_nounit = spectrum
        spectrum_unit = 1

    # Compute pixel resolution R_pix_vec
    data_waves_diff = np.diff(data_waves, prepend=2 * data_waves[0] - data_waves[1])
    R_pix_vec = data_waves.to(u.nm).value / data_waves_diff.to(u.nm).value
    R_pix_vec = np.interp(waves.to(u.nm).value,
                                data_waves.to(u.nm).value,
                                R_pix_vec)

    if broaden_pixel:
        # Broaden to instrumental resolution
        R_vec = np.full_like(waves.value, R)
        spectrum_broadR = broaden(waves, spectrum_nounit, R_vec, kernel="gaussian",n_jobs=n_jobs)

        # Broaden to pixel resolution
        spectrum_broadpix = broaden(waves, spectrum_broadR, R_pix_vec, kernel="box",n_jobs=n_jobs,wave_samples_only =data_waves)
    else:
        # Broaden to instrumental resolution
        # This assumes that the pixel broadening is included in the resolution value.
        R_vec = np.full_like(waves.value, R)
        spectrum_broadpix = broaden(waves, spectrum, R_vec, kernel="gaussian",n_jobs=n_jobs,wave_samples_only =data_waves)

    # Resample
    spectrum_resamp = np.interp(data_waves.to(u.nm).value,
                                waves.to(u.nm).value,
                                spectrum_broadpix)

    # import matplotlib.pyplot as plt
    # plt.title("Spectral templates")
    # plt.plot(waves.to(u.nm).value, spectrum, label="spectrum_resamp",color="red")
    # plt.plot(data_waves.to(u.nm).value, spectrum_resamp, label="spectrum_resamp",color="orange")
    # plt.plot(waves.to(u.nm).value, spectrum_broadpix, "--", label="spectrum_broadpix",color="blue")
    # plt.scatter(waves.to(u.nm).value, spectrum_broadpix,c="blue")
    # plt.show()
    return spectrum_resamp*spectrum_unit




def find_bracketing_indices(wave_original, wave_samples):
    """
    Return a set of unique indices from wave_original that bracket the wave_samples.

    Parameters
    ----------
    wave_original : array-like
        A 1D array of increasing wavelength values.
    wave_samples : array-like
        A 1D array of wavelength samples to bracket.

    Returns
    -------
    unique_indices : set
        Set of unique indices i such that wave_original[i] <= sample < wave_original[i+1]
        (both i and i+1 are included in the result set). Samples outside the bounds
        of wave_original are ignored.
    """
    wave_original = np.asarray(wave_original)
    wave_samples = np.asarray(wave_samples)

    if not np.all(np.diff(wave_original) > 0):
        raise ValueError("wave_original must be strictly increasing")

    inds = np.searchsorted(wave_original, wave_samples, side='right') - 1
    valid = (inds >= 0) & (inds < len(wave_original) - 1)

    inds_valid = inds[valid]
    all_indices = np.concatenate([inds_valid, inds_valid + 1])

    return all_indices



def downsample_spectrum(wave0, spec0, new_wave):
    """
    Downsamples a spectrum by summing flux within bins defined by new_wave.

    Parameters
    ----------
    wave0 : np.ndarray
        Original wavelength array (must be sorted).
    spec0 : np.ndarray
        Original spectrum corresponding to wave0.
    new_wave : np.ndarray
        New wavelength bin centers.

    Returns
    -------
    new_spec : np.ndarray
        Downsampled spectrum: sum of spec0 within each bin.
        Returns full of zeros if np.size(new_wave) < np.size(wave0).
    """

    if isinstance(spec0, Quantity):
        spec0_nounit = spec0.value
        spec0_unit = spec0.unit
    else:
        spec0_nounit = spec0
        spec0_unit = 1

    if np.size(new_wave) > np.size(wave0):
        return np.zeros_like(new_wave)*spec0_unit

    # Define bin edges from midpoints between new_wave values
    bin_edges = np.zeros(len(new_wave) + 1)
    bin_edges[1:-1] = (new_wave[:-1] + new_wave[1:]) / 2
    bin_edges[0] = new_wave[0] - (new_wave[1] - new_wave[0]) / 2
    bin_edges[-1] = new_wave[-1] + (new_wave[-1] - new_wave[-2]) / 2

    # Assign each wave0 point to a bin
    inds = np.digitize(wave0, bin_edges) - 1  # shift to 0-based index

    # Use bincount to sum spec0 in each bin
    valid = (inds >= 0) & (inds < len(new_wave))
    new_spec = np.bincount(inds[valid], weights=spec0_nounit[valid], minlength=len(new_wave))

    return new_spec*spec0_unit

def regularized_inverse(cov, threshold=1e-10):
    """
    Computes a regularized inverse of a covariance matrix by thresholding small/negative eigenvalues.

    Parameters
    ----------
    cov : ndarray
        The covariance matrix to invert.
    threshold : float
        Eigenvalue threshold below which components are clamped.

    Returns
    -------
    inv_cov : ndarray
        The regularized inverse of the covariance matrix.
    reg_cov : ndarray
        The regularized covariance matrix (with eigenvalues clamped).
    """
    # Eigen-decomposition (covariance matrix is symmetric)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Clamp eigenvalues to the minimum threshold
    eigvals_clamped = np.clip(eigvals, np.nanmax(eigvals)*threshold, None)

    # Reconstruct regularized covariance matrix
    reg_cov = eigvecs @ np.diag(eigvals_clamped) @ eigvecs.T

    # Reconstruct inverse using clamped eigenvalues
    inv_cov = eigvecs @ np.diag(1.0 / eigvals_clamped) @ eigvecs.T

    return inv_cov, reg_cov,eigvecs



def write_snr_results_to_file(results_dict, output_filename, config_json_path=None):
    """
    Write a machine-readable summary of SNR analysis results to a structured text file.

    This function formats and writes the contents of a results dictionary to a tab-delimited
    text file. The file includes:

    - A commented header section describing the meaning of each column.
    - A table where each row corresponds to a star and each column to a specific SNR metric.
    - A commented copy of the JSON configuration file used to generate the results,
      appended at the end for reproducibility.

    Parameters
    ----------
    results_dict : dict
        Dictionary containing the SNR results. Must include a key 'stars' (array of star names)
        and any number of SNR keys with NumPy arrays of the same length as 'stars'.
        Expected key pattern: 'SNR_<mol>_<type>'.

    config_json_path : str
        Path to the JSON configuration file used to generate the results. This will be appended
        at the end of the output file as a commented block for reference.

    output_filename : str
        Path to the output text file where the results will be written. Existing files
        with the same name will be overwritten.
    """
    snr_keys = [key for key in results_dict.keys() if "stars" not in key]

    descriptions = {
        "star": "The name of the stars corresponding to each SNR values.",
        "SNR_<mol>_avg_per_bin": "SNR per spectral bin",
        "SNR_<mol>_ignore_corr": "Broadband SNR ignoring any correlations in the data.",
        "SNR_<mol>_corr": "Broadband SNR accounting for the covariance.",
        "SNR_<mol>_uncorr_small_scale": "SNR including features smaller than correlation length (high-pass)",
        "SNR_<mol>_corr_large_scale": "SNR including features larger than correlation length (low-pass)",
        "SNR_<mol>_corr_test": "Quadrature sum of small and large scale SNRs"
    }

    # Define field widths
    star_width = max(len(s) for s in results_dict["stars"]) + 2
    snr_width = max(len(s) for s in snr_keys) + 2  # Enough for header + float formatting

    with open(output_filename, 'w') as f:
        # Write header comments
        f.write("# SNR output table\n")
        f.write("# Column descriptions:\n")
        for key in descriptions.keys():
            f.write(f"# {key}: {descriptions[key]}\n")
        f.write("#\n")

        # Write column headers
        header = ["star".ljust(star_width)] + [key.ljust(snr_width) for key in snr_keys]
        f.write("\t".join(header) + "\n")

        # Write data rows
        stars = results_dict["stars"]
        for i in range(len(stars)):
            row = [stars[i].ljust(star_width)]
            for key in snr_keys:
                val = results_dict[key][i]
                val_str = "nan" if np.isnan(val) else f"{val:.2f}"
                row.append(val_str.rjust(snr_width))
            f.write("\t".join(row) + "\n")

        if config_json_path is not None:
            # Append JSON config at the end as comments
            f.write("\n# Configuration parameters used:\n")
            with open(config_json_path, "r") as cf:
                config_lines = cf.readlines()
                for line in config_lines:
                    f.write("# " + line)


def read_snr_results_from_file(filename):
    """
    Read a structured tab-delimited SNR output text file and reconstruct the original results dictionary.

    This function parses a file generated by `write_snr_results_to_file`. It automatically extracts:

    - Column headers from the first non-comment line
    - Star names from the first column of each row (supports names with spaces)
    - SNR values from the remaining columns, converting "nan" to `np.nan` and numeric strings to floats

    The function skips:
    - All comment lines starting with '#'
    - The JSON configuration block at the end of the file

    Parameters
    ----------
    filename : str
        Path to the SNR output text file.

    Returns
    -------
    results_dict : dict
        Dictionary with keys:
        - 'stars' : NumPy array of star names (dtype=str)
        - one key per SNR metric, each mapping to a NumPy array of float values

    Raises
    ------
    ValueError
        If no valid data lines are found in the file.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Skip comment lines and find the header
    header_line = next(line for line in lines if not line.strip().startswith("#"))
    headers = header_line.strip().split("\t")
    snr_keys = headers[1:]  # exclude "star"
    snr_keys = [key.strip() for key in snr_keys]

    results_dict = {"stars": []}
    for key in snr_keys:
        results_dict[key] = []

    for line in lines:
        if line.strip().startswith("#") or not line.strip() or line.strip().startswith("{"):
            continue  # Skip comments, blank lines, or start of config
        if line == header_line:
            continue  # Already parsed header

        fields = line.strip().split("\t")
        if len(fields) < len(headers):
            continue  # Incomplete row or junk at the bottom

        # First column is star name (can have spaces), rest are floats or "nan"
        results_dict["stars"].append(fields[0].strip())
        for key, val in zip(snr_keys, fields[1:]):
            results_dict[key].append(np.nan if val.strip() == "nan" else float(val))

    # Convert lists to numpy arrays
    results_dict["stars"] = np.array(results_dict["stars"], dtype=str)
    for key in snr_keys:
        results_dict[key] = np.array(results_dict[key], dtype=float)

    return results_dict

import json
def read_snr_results_and_json_from_file(filename):
    """
    Read a structured tab-delimited SNR output text file and reconstruct the original results dictionary
    and a trailing JSON configuration block written as commented lines.

    Parameters
    ----------
    filename : str
        Path to the SNR output text file.

    Returns
    -------
    results_dict : dict
        Dictionary with keys:
        - 'stars' : NumPy array of star names (dtype=str)
        - one key per SNR metric, each mapping to a NumPy array of float values

    config_dict : dict
        Dictionary parsed from the JSON block at the end of the file (if present), or empty dict.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Identify the header line (first non-comment)
    header_line = next(line for line in lines if not line.strip().startswith("#"))
    headers = header_line.strip().split("\t")
    snr_keys = [key.strip() for key in headers[1:]]  # skip 'star' column

    results_dict = {"stars": []}
    for key in snr_keys:
        results_dict[key] = []

    data_started = False
    json_lines = []

    for line in lines:
        if line.strip().startswith("#") or not line.strip() or line.strip().startswith("{"):
            continue  # Skip comments, blank lines, or start of config
        if line == header_line:
            continue  # Already parsed header

        fields = line.strip().split("\t")
        if len(fields) < len(headers):
            continue  # Incomplete row or junk at the bottom

        # First column is star name (can have spaces), rest are floats or "nan"
        results_dict["stars"].append(fields[0].strip())
        for key, val in zip(snr_keys, fields[1:]):
            results_dict[key].append(np.nan if val.strip() == "nan" else float(val))

    for line in lines:
        if not data_started:
            if not line.startswith("#"):
                data_started = True
            continue  # skip pre-header comments

        # After table ends, collect JSON-comment lines
        if line.startswith("#"):
            json_lines.append(line.lstrip("#"))
        continue


    # Convert lists to arrays
    results_dict["stars"] = np.array(results_dict["stars"], dtype=str)
    for key in snr_keys:
        results_dict[key] = np.array(results_dict[key], dtype=float)

    # Parse JSON block
    config_dict = {}
    if json_lines:
        try:
            json_str = "".join(json_lines[1::])
            config_dict = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse JSON block: {e}")

    return results_dict, config_dict