import matplotlib.pyplot as plt
from maria import Band

my_band = Band(center=150e9, # in Hz
               width=30e9) # in Hz

my_band.plot()