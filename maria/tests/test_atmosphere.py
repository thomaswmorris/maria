import pytest
import maria
from maria import models

@pytest.mark.atmosphere
def test_ACT_PA4_LAM():

    lam = models.LinearAngularModel(array=maria.get_array('ACT_PA4'), 
                                    pointing=maria.get_pointing('STARE_0az_90el_60s'), 
                                    site=maria.get_site('ACT'), 
                                    verbose=True)

    lam.simulate_temperature(units='K_RJ')
    assert (lam.temperature > 0).all()

@pytest.mark.atmosphere
def test_GBT_MUSTANG2_LAM():

    lam = models.LinearAngularModel(array=maria.get_array('MUSTANG-2'), 
                                    pointing=maria.get_pointing('DAISY_2deg_4ra_10dec_600s'), 
                                    site=maria.get_site('GBT'), 
                                    verbose=True)

    lam.simulate_temperature(units='K_RJ')
    assert (lam.temperature > 0).all()


@pytest.mark.atmosphere
def test_JCMT_SCUBA2_LAM():

    lam = models.LinearAngularModel(array=maria.get_array('SCUBA-2'), 
                                    pointing=maria.get_pointing('BAF_5deg_45az_45el_60s'), 
                                    site=maria.get_site('JCMT'), 
                                    verbose=True)

    lam.simulate_temperature(units='K_RJ')
    assert (lam.temperature > 0).all()