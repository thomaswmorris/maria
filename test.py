from maria import * 
import matplotlib.pyplot as plt

for file in ['./maps/tsz.fits', './maps/tsz_x.fits']:
    obs = Weobserve(
        project       = './Mock_obs',
        skymodel      = file, 
        verbose       = True,

        inbright      = -5.37 * 1e7 * 0.00011347448463627645,
        incell        = 1/360 #degree
        )