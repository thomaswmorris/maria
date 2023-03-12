import pytest
import maria 

ACT_config = { 
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
             }


daisy_config =  {
        'start_time'   : '2022-07-01T08:00:00',
        'end_time'     : '2022-07-01T08:05:00',
        'scan_pattern' : 'daisy',          # [.]   the type of scan strategy (SS)
        'scan_options' : {'k' : 3.1416},   # 
        'coord_center' : (4, 10.5),        # are given in deg in the coord_frame input
        'coord_throws' : (2, 2),
        'coord_frame'  : 'ra_dec',
        'coord_units'  : 'degrees',
        'scan_period'  : 60,               # [s]   how often the scan pattern repeats
        'sample_rate'  : 20,               # [Hz]  how fast to sample
        'seed'         : 42
                }


@pytest.fixture(scope="function")
def ACT_array():
    """
    Returns a maria.Array() object for ACT
    """
    return maria.Array(ACT_config)


@pytest.fixture(scope="function")
def daisy_scan():
    """
    Returns a maria.Pointing() object for a daisy scan
    """
    return maria.Pointing(config=daisy_config)


@pytest.fixture(scope="function")
def ACT_site():
    """
    Returns a maria.Site() object for Chajnantor
    """
    return maria.Site(region='chajnantor', altitude=5190)