DEFAULT_LAM_CONFIG = {'min_depth' : 500,
                      'max_depth' : 2000,
                       'n_layers' : 3,
                       'min_beam_res' : 8,
                       }

OBSERVATORIES = {
    'AtLAST': {
        'site'           : 'chajnantor',
        'bands'          : [(27e9, 5e9, 100),      # (band center, bandwidth, dets per band) [GHz, GHz, .]
                            (39e9, 5e9, 100), 
                            (93e9, 10e9, 100), 
                            (145e9, 10e9, 100), 
                            (225e9, 30e9, 100), 
                            (280e9, 40e9, 100)],     
        'geometry'      : 'hex',                   # [.]   type of detector distribution
        'field_of_view' : 1.3,                     # [deg] maximum det separation
        'primary_size'  : 50,                      # [m]   size of the primary mirror
        'band_grouping' : 'randomized',            # [.]   type of band distribution --> other options would be: 
        'az_bounds'     : [0, 360],                # [.]   type of band distribution
        'el_bounds'     : [20, 90],
        'max_az_vel'    : 3,            
        'max_el_vel'    : 2,
        'max_az_acc'    : 1,
        'max_el_acc'    : 0.25,
    },

    'ACT': { 
        'site'          : 'chajnantor',
        'bands'         : [(27e9, 10e9, 50), 
                           (39e9, 10e9, 50),
                           (93e9, 10e9, 50), 
                           (145e9, 10e9, 50), 
                           (225e9, 10e9, 50), 
                           (280e9, 10e9, 50)],     # [Hz]  band centers
        'geometry'      : 'hex',                   # [.]   type of detector distribution
        'field_of_view' :  0.5,                    # [deg] maximum det separation
        'primary_size'  :  12,                     # [m]   size of the primary mirror
        'band_grouping' : 'randomized', 
        'az_bounds'     : [0, 360],                # [.]   type of band distribution
        'el_bounds'     : [20, 90],
        'max_az_vel'    : 3,            
        'max_el_vel'    : 2,
        'max_az_acc'    : 1,
        'max_el_acc'    : 0.25,
    },

    'GBT': { 
        ...
    }
}

SCANNINGPATTERNS = {
    'static': {
        'start_time'    : '2022-07-01T08:00:00',
        'end_time'      : '2022-07-01T08:10:00',
        'scan_pattern'  : 'back-and-forth',
        'scan_options'  : {'k' : 3.1416},      # 
        'coord_center'  : (4, 10.5),           # are given in deg in the coord_frame input
        'coord_throw'   : (2, 0),
        'coord_frame'   : 'ra_dec',            # ra_dec or az_el
        'scan_period'   : 120,                 # [s]   how often the scan pattern repeats
        'sample_rate'   : 20,                  # [Hz]  how fast to sample
    },

    'box': {
        ...    
    },

    'daisy': {
        'start_time'   : '2022-07-01T08:00:00',
        'end_time'     : '2022-07-01T08:20:00',
        'scan_pattern' : 'daisy',          # [.]   the type of scan strategy (SS)
        'scan_options' : {'k' : 3.1416},   # 
        'coord_center' : (4, 10.5),        # are given in deg in the coord_frame input
        'coord_throw'  : (2, 2),
        'coord_frame'  : 'ra_dec',
        'scan_period'  : 60,               # [s]   how often the scan pattern repeats
        'sample_rate'  : 20,               # [Hz]  how fast to sample
    },

    'lissajous': {
        ...
    },
    
    'rose': {
        ...
    }

}
