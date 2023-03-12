import maria
from maria import models


def test_linear_angular_model(ACT_array, daisy_scan, ACT_site):

    lam = models.LinearAngularModel(array=ACT_array, pointing=daisy_scan, site=ACT_site, verbose=True)
    lam.simulate_integrated_water_vapor()

    assert (lam.integrated_water_vapor > 0).all()
