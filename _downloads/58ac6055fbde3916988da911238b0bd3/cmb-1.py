from maria.cmb import generate_cmb_patch

cmb_patch = generate_cmb_patch(width=5) # in degrees

cmb_patch.plot(cmap="cmb")