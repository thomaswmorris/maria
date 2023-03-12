import pytest
import maria
from maria import models

@pytest.mark.atmosphere
def ACT_LAM():

    lam = models.LinearAngularModel(array=maria.get_array('ACT_PA5'), 
                                    pointing=maria.get_pointing('DAISY_5deg_45az_45el_60s'), 
                                    site=maria.get_site('ACT'), 
                                    verbose=True)

    lam.simulate_integrated_water_vapor()

    assert (lam.integrated_water_vapor > 0).all()

@pytest.mark.atmosphere
def GBT_LAM():

    lam = models.LinearAngularModel(array=maria.get_array('MUSTANG2'), 
                                    pointing=maria.get_pointing('DAISY_5deg_45az_45el_60s'), 
                                    site=maria.get_site('GBT'), 
                                    verbose=True)

    lam.simulate_integrated_water_vapor()

    assert (lam.integrated_water_vapor > 0).all()
