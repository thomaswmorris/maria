import maria

map_filename = maria.io.fetch("maps/big_cluster.fits")

input_map = maria.map.read_fits(filename=map_filename,
                                index=1, # which index of the HDU to read
                                nu=150., # in GHz
                                resolution=1/1024,
                                center=(150, 10),
                                frame="ra_dec",
                                units="Jy/pixel")

input_map.to(units="uK_RJ").plot()