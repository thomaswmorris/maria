Customizing arrays
++++++++++++++++++

The ``array`` determines the characteristics of the observing instrument. We can make an array with::

    my_array = maria.get_array('AtLAST')



Simulations are made up of an array (defining the instrument), a site (where on earth it is), and a pointing (defining how it observes). Simulations for a few different telescopes might be instantiated as::

    # MUSTANG-2
    mustang2_sim = Simulation(array='MUSTANG-2',
                              site='green_bank',
                              pointing='daisy',
                              map_file=path_to_some_fits_file,  # Input files must be a fits file.
                              map_units='Jy/pixel',  # Units of the input map in Kelvin Rayleigh Jeans (K, defeault) or Jy/pixel
                              map_res=pixel_size,
                              map_center=(10, 4.5),  # RA & Dec. in degrees
                              map_freqs=[90],
                              degrees=True)
