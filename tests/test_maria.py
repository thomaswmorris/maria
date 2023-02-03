import maria 
from maria import utils

AtLAST_config = {           'site' : 'chajnantor',
                           'bands' : [(27e9, 10e9, 50),   # (band center, bandwidth, dets per band) [GHz, GHz, .]
                                      (39e9, 10e9, 50), 
                                      (93e9, 10e9, 50), 
                                      (145e9, 10e9, 50), 
                                      (225e9, 10e9, 50), 
                                      (280e9, 10e9, 50)],     
                        'geometry' : 'hex',               # [.]   type of detector distribution
                   'field_of_view' : 0.5,                 # [deg] maximum det separation
                    'primary_size' : 50,                  # [m]   size of the primary mirror
                   'band_grouping' : 'randomized',        # [.]   type of band distribution
                       'az_bounds' : [0, 360],            # [.]   type of band distribution
                       'el_bounds' : [30, 90],
                      'max_az_vel' : 3,            
                      'max_el_vel' : 2,
                      'max_az_acc' : 1,
                      'max_el_acc' : 0.25,
                }

static_config = { 'coord_throw' : (2, 0),
                 'scan_options' : {'k' : 3.1416}, # 
                 'scan_pattern' : 'back-and-forth',
                  'scan_period' : 120,        # [s]   how often the scan pattern repeats
                  'sample_rate' : 20,        # [Hz]  how fast to sample
              }

from maria.utils import Planner 

AtLAST  = maria.Array(AtLAST_config)

planner = Planner(AtLAST)
plan_configs = planner.make_plans(start='2023-08-01', 
                                  end='2023-08-02', 
                                  ra=45, 
                                  dec=-9, 
                                  chunk_time=600,
                                  static_config=static_config)

plan_config_list = list(plan_configs)

plan = maria.Plan(plan_config_list[0])

# @pytest.mark.parametrize()
def test_linear_angular_model():
    
    lam = maria.LAM(AtLAST, plan, verbose=True) 
    lam.simulate()