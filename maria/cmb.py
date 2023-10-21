# class CMBSimulation(base.BaseSimulation):
#     """
#     This simulates scanning over celestial sources.
#     """
#     def __init__(self, array, pointing, site, map_file, add_cmb=False, **kwargs):
#         super().__init__(array, pointing, site)

#         pass

#     def _get_CMBPS(self):

#         import camb

#         pars = camb.CAMBparams()
#         pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
#         pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)

#         # correct mode would l=129600 for 5"
#         pars.set_for_lmax(5000, lens_potential_accuracy=0)

#         results = camb.get_results(pars)
#         powers = results.get_cmb_power_spectra(pars, CMB_unit="K")["total"][:, 0]

#         self.CMB_PS = np.empty((len(self.array.ubands), len(powers)))
#         for i in range(len(self.array.ubands)):
#             self.CMB_PS[i] = powers


#     def _cmb_imager(self, bandnumber=0):

#         import pymaster as nmt

#         nx, ny = self.map_data[0].shape
#         Lx = nx * np.deg2rad(self.sky_data["incell"])
#         Ly = ny * np.deg2rad(self.sky_data["incell"])

#         self.CMB_map = nmt.synfast_flat(
#             nx,
#             ny,
#             Lx,
#             Ly,
#             np.array([self.CMB_PS[bandnumber]]),
#             [0],
#             beam=None,
#             seed=self.pointing.seed,
#         )[0]

#         self.CMB_map += utils.Tcmb
#         self.CMB_map *= utils.KcmbToKbright(np.unique(self.array.dets.band_center)[bandnumber])

#     #self._cmb_imager(i)
#         #         cmb_data = sp.interpolate.RegularGridInterpolator(
#         #                     (map_x, map_y), self.CMB_map, bounds_error=False, fill_value=0
#         #                     )((lam_x, lam_y))
#         #         self.noise    = self.lam.temperature + cmb_data
#         #         self.combined = self.map_data + self.lam.temperature + cmb_data
