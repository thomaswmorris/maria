Simulate
============

In this section, we provide a brief explanation of how to create synthetic time streams using Maria. For a comprehensive overview, please refer to the Tutorials.

The following few lines show how to generate the time streams in Python::

    from maria import Simulation
    sim = Simulation(array='MUSTANG-2', pointing='daisy', site='GBT', atm_model='single_layer', map_file=inputfile, map_res=pixel_size)
    tod = sim.run()

To make a map out of the TOD, simply run::

    from maria import mappers
    import numpy as np

    mapper = mappers.BinMapper(map_height = map_size, #radians
                              map_width   = map_size,  #radians
                              map_res     = np.radians(pixel_size),  #radians
                              map_filter  = True,
                              n_modes_to_remove = 1)
    mapper.add_tods(tod)
    mapper.run()
    mapper.save_maps("output.fits")