from maria import * 
import matplotlib.pyplot as plt

obs = Weobserve(
    project       = './Mock_obs',
    skymodel      = './maps/tsz.fits', 
    verbose       = True,

    inbright      = -5.37 * 1e7 * 0.00011347448463627645,
    incell        = 1/3600 #degree
    )

obs = Weobserve(
    project       = './Mock_obs',
    skymodel      = './maps/tsz_x.fits', 
    verbose       = True,

    inbright      = -5.37 * 1e7 * 0.00016408804725041452,
    incell        = 1/3600 #degree
    )