import maria
from maria import models


def ACT_daisy_scan():

    lam = models.LinearAngularModel(array=maria.get_array('ACT'), 
                                    pointing=maria.get_pointing('DAISY_2deg_4ra_10.5dec_600s'), 
                                    site=maria.get_site('ACT'), 
                                    verbose=True)

    lam.simulate_integrated_water_vapor()

    assert (lam.integrated_water_vapor > 0).all()
