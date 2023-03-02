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