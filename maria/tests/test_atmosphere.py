import pytest
import maria

# @pytest.mark.atmosphere
# def test_ACT_PA4_LAM():

#     lam = atmosphere.LinearAngularModel(array=maria.get_array('ACT_PA4'), 
#                                     pointing=maria.get_pointing('STARE_0az_90el_60s'), 
#                                     site=maria.get_site('ACT'), 
#                                     verbose=True)

#     lam.simulate_temperature(units='K_RJ')
#     assert (lam.temperature > 0).all()

@pytest.mark.atmosphere
def test_GBT_MUSTANG2_LAM():

    mustang_2 = maria.get_array("MUSTANG-2")
    stare = maria.get_pointing("STARE_0az_90el_60s")
    green_bank = maria.get_site("GBT")

    sim = maria.Simulation(array=mustang_2, pointing=stare, site=green_bank)

    tod = sim.run()


# @pytest.mark.atmosphere
# def test_JCMT_SCUBA2_LAM():

#     lam = atmosphere.LinearAngularModel(array=maria.get_array('SCUBA-2'), 
#                                     pointing=maria.get_pointing('BAF_5deg_45az_45el_60s'), 
#                                     site=maria.get_site('JCMT'), 
#                                     verbose=True)

#     lam.simulate_temperature(units='K_RJ')
#     assert (lam.temperature > 0).all()