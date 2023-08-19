import numpy as np

from .tod import TOD

from .base import BaseSimulation

from .array import get_array, get_array_config
from .pointing import get_pointing, get_pointing_config
from .site import get_site, get_site_config

from . import atmosphere, cmb, noise, sky


class Simulation(BaseSimulation):
    """
    A simulation! This is what users should touch, primarily. 
    """
    def __init__(self, 
                 array, 
                 pointing, 
                 site, 
                 atm_model="linear_angular", 
                 map_file=None, 
                 noise_model="white", 
                 **kwargs):

        if isinstance(array, str):
            array = get_array(array, **kwargs)

        if isinstance(pointing, str):
            pointing = get_pointing(pointing, **kwargs)

        if isinstance(site, str):
            site = get_site(site, **kwargs)

        # json_pattern = os.path.join(here+'/configs', '*.json')
        # json_files    = glob.glob(json_pattern)
        # test_multiple_json_files(json_files, *tuple(kwargs.keys()))

        super().__init__(array, pointing, site)

        self.atm_model = atm_model
        if atm_model in ["linear_angular", "LA"]:
            self.atm_sim = atmosphere.LinearAngularSimulation(array, pointing, site, **kwargs)
        elif atm_model in ["kolmogorov_taylor", "KT"]:
            self.atm_sim = atmosphere.KolmogorovTaylorSimulation(array, pointing, site, **kwargs)
        else:
            self.atm_sim = None

        self.map_file = map_file
        if map_file is not None:
            self.map_sim = sky.MapSimulation(array, pointing, site, map_file, **kwargs)
        else:
            self.map_sim = None

    def run(self):

        if self.atm_sim is not None:
            self.atm_sim.run()

        if self.map_sim is not None:
            self.map_sim.run()

        tod = TOD()

        tod.time = self.pointing.unix
        tod.az   = self.pointing.az
        tod.el   = self.pointing.el
        tod.ra   = self.pointing.ra
        tod.dec  = self.pointing.dec

        # number of bands are lost here
        tod.data = np.zeros((self.array.n_dets, self.pointing.n_time))

        if self.atm_sim is not None:
            tod.data += self.atm_sim.temperature

        if self.map_sim is not None:
            tod.data += self.map_sim.temperature

        tod.detectors = self.array.dets

        tod.metadata = {'latitude': self.site.latitude,
                        'longitude': self.site.longitude,
                        'altitude': self.site.altitude}

        return tod

    def save_maps(self, mapper):

            self.map_sim.map_data["header"]['comment'] = 'Made Synthetic observations via maria code'
            self.map_sim.map_data["header"]['comment'] = 'Changed resolution and size of the output map'
            self.map_sim.map_data["header"]['CDELT1']  = np.rad2deg(mapper.resolution)
            self.map_sim.map_data["header"]['CDELT2']  = np.rad2deg(mapper.resolution)
            self.map_sim.map_data["header"]['CRPIX1']  = mapper.maps[list(mapper.maps.keys())[0]].shape[0]
            self.map_sim.map_data["header"]['CRPIX2']  = mapper.maps[list(mapper.maps.keys())[0]].shape[1]

            self.map_sim.map_data["header"]['comment'] = 'Changed pointing location of the output map'
            self.map_sim.map_data["header"]['CRVAL1']  = self.pointing.ra.mean()
            self.map_sim.map_data["header"]['CRVAL2']  = self.pointing.dec.mean()

            self.map_sim.map_data["header"]['comment'] = 'Changed spectral position of the output map'
            self.map_sim.map_data["header"]['CTYPE3']  = 'FREQ    '
            self.map_sim.map_data["header"]['CUNIT3']  = 'Hz      '
            self.map_sim.map_data["header"]['CRPIX3']  = 1.000000000000E+00
            
            self.map_sim.map_data["header"]['BTYPE']   = 'Intensity'
            if self.map_sim.map_data["inbright"] == 'Jy/pixel': 
                self.map_sim.map_data["header"]['BUNIT']   = 'Jy/beam '   
            else: 
                self.map_sim.map_data["header"]['BUNIT']   = 'Kelvin RJ'   

            for i in range(len(self.array.ubands)):
                
                self.map_sim.map_data["header"]['CRVAL3']  = self.array.detectors[i][0]
                self.map_sim.map_data["header"]['CDELT3']  = self.array.detectors[i][1]

                save_map = mapper.maps[list(mapper.maps.keys())[i]] 

                if self.map_sim.map_data["inbright"] == 'Jy/pixel': 
                    save_map *= utils.KbrightToJyPix(self.map_data["header"]['CRVAL3'], 
                                                     self.map_data["header"]['CDELT1'], 
                                                     self.map_data["header"]['CDELT2']
                                                    )
                fits.writeto( filename = here+'/outputs/atm_model_'+str(self.atm_model)+'_file_'+ str(self.map_file.split('/')[-1].split('.fits')[0])+'.fits', 
                                  data = save_map, 
                                header = self.map_sim.map_data["header"],
                             overwrite = True 
                            )

