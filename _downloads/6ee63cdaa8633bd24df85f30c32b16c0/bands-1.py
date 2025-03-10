import matplotlib.pyplot as plt
from maria import Band

my_band = Band(center=150, # in GHz
               width=30) # in GHz

my_band.plot()