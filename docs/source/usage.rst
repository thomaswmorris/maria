Usage
#####

Running simulations with maria looks like::

    import maria

    sim = maria.Simulation(**keywords)

    tod = sim.run() # running a simulation returns a ``TOD'' object

    another_tod = sim.run() # simulations can be run any number of times

Simulations are made up of an array (defining the instrument), a pointing (defining how it collects data), and a site (where on earth it is). For example:::

    mustang_sim = maria.Simulation(array="MUSTANG-2", pointing="daisy", site="GBT")

    act_sim = maria.Simulation(array="ACT_PA5", pointing="back-and-forth", site="ACT")

    spt_sim = maria.Simulation(array="SPT", pointing="stare", site="ACT")

We can also specify each parameter of the simulation separately ::

    sim = maria.Simulation(
        # Mandatory minimal weither settings
        # ---------------------
        array="MUSTANG-2",  # Array type
        pointing="daisy",  # Scanning strategy
        site="GBT",  # Site
        atm_model="single_layer",  # The atmospheric model, set to None if you want a noiseless observation.
        # True sky input
        # ---------------------
        map_file="maps/cluster.fits",  # Input files must be a fits file.
        # map_file can also be set to None if are only interested in the noise
        map_center=(150.0, 10),  # RA & Dec in degree
        # Defeault Observational setup
        # ----------------------------
        integration_time=600,  # seconds
        scan_center=(150.0, 10),  # degrees
        pointing_frame="ra_dec",  # frame
        scan_options={"radius": 0.05, "speed": 0.05, "petals": 5},
        # Additional inputs:
        # ----------------------
        map_units="Jy/pixel",  # Kelvin Rayleigh Jeans (K, defeault) or Jy/pixel
        # map_inbright = -6e-6,                        # Linearly scale the map to have this peak value.
        map_res=0.1 / 1000,  # degree, overwrites header information
    )
