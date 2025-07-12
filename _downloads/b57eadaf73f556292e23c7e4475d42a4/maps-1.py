import maria

map_filename = maria.io.fetch("maps/einstein.h5")

input_map = maria.map.load(filename=map_filename,
                           nu=150e9, # in Hz
                           resolution=1/1024,
                           center=(150, 10),
                           frame="ra_dec",
                           units="Jy/pixel")

print(input_map)
input_map.to(units="uK_RJ").plot()