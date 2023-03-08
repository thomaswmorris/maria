from maria import * 
import matplotlib.pyplot as plt

for file in ['./maps/tsz.fits']:
    obs = Weobserve(
        project       = './Mock_obs',
        skymodel      = file, 
        verbose       = True,
        
        bands         = [(27e9, 5e9, 100)],      # (band center, bandwidth, dets per band) [GHz, GHz, .]
        inbright      = -5.37 * 1e4 * 0.00011347448463627645,
        incell        = 1/360 #degree
        )