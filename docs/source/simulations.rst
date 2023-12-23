Usage
+++++

We simulate observations by defining a ``Simulation`` object and running it to produce a ``TOD`` object::

    from maria import Simulation

    sim = maria.Simulation()

    tod = sim.run()


The same simulation can produce any number of ``TOD`` objects, continuing from where the last TOD left off::

    another_tod = sim.run()

    yet_another_tod = sim.run()


Customizing simulations
+++++++++++++++++++++++

Simulations are made up of an ``array`` (defining the instrument), a ``site`` (where on earth it is), and a ``pointing`` (defining how it observes). Simulations for a few different telescopes might be instantiated as::

    # The Atacama Cosmology Telescope
    act_sim = Simulation(array='ACT', site='cerro_toco', pointing='back-and-forth')

    # MUSTANG-2
    mustang2_sim = Simulation(array='MUSTANG-2', site='green_bank', pointing='daisy')

    # The South Pole Telescope
    spt_sim = Simulation(array='SPT', site='amundsen_scott', pointing='stare')

Supported arrays are ``maria.all_arrays``


We can think of

We can also pass other arguments to the ``Simulation``, which will overwrite the values it takes


    # The Atacama Cosmology Telescope
    act_sim = Simulation(array='ACT', site='cerro_toco', pointing='back-and-forth')

    # MUSTANG-2
    mustang2_sim = Simulation(array='MUSTANG-2', site='green_bank', pointing='daisy')

    # The South Pole Telescope
    spt_sim = Simulation(array='SPT', site='amundsen_scott', pointing='stare')


All the supported values ``array``, ``pointing`` and ``site`` can be found with::

    print(f'Supported arrays: {maria.all_arrays}')
    print(f'Supported sites: {maria.all_sites}')
    print(f'Supported pointings: {maria.all_pointings}')

Map simulations
+++++++++++++++

We can simulate observing (and later mapping) some celestial signal by supplying the simulation with a map to use as the ground truth.



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

    # The South Pole Telescope
    spt_sim = Simulation(array='SPT', site='amundsen_scott', pointing='stare')





We can also specify each parameter of the simulation separately ::

    sim = maria.Simulation(
                array='MUSTANG-2',
                pointing='daisy',
                site='green_bank',
                atmosphere_model='2d',
                array=# defaults to a small test array
        array_description:
        detector_config:
            f093:
            n=60
            band_center=90
            band_width=10
            f150:
            n=60
            band_center=150
            band_width=20
        field_of_view=0.8
        baseline=0
        geometry=0
        primary_size=6

        start_time=2022-02-10T06:00:00
        integration_time=60 # in seconds
        pointing_frame=az_el
        degrees=True
        sample_rate=20
        scan_pattern=daisy
        scan_center=(10, 4.5)
        scan_options={}

        region='chajnantor'
        latitude=-23.0294
        longitude=-67.7548
        altitude=5064
        site_documentation=''
        weather_quantiles={}

        atmosphere_model='2d'
        min_atmosphere_beam_res=4
        min_atmosphere_height=500
        max_atmosphere_height=5000
        turbulent_outer_scale=800
        pwv_rms_frac=0.03
        pwv=2

        map_file=''
        map_frame=ra_dec
        map_center=[10, 4.5]
        map_res=0.5
        map_inbright:
        map_units=K_RJ
        map_freqs=[150]

        white_noise_level=1.e-2 # in Kelvin Rayleigh-Jeans equivalent
        pink_noise_level=1.e-2 # in Kelvin Rayleigh-Jeans equivalent amplitude in fourier domain
        pink_noise_slope=0.5

            )



We can also specify each parameter of the simulation separately ::

    sim = maria.Simulation(
                array='MUSTANG-2',
                pointing='daisy',
                site='green_bank',
                atmosphere_model='2d',
                array=# defaults to a small test array
        array_description:
        detector_config:
            f093:
            n=60
            band_center=90
            band_width=10
            f150:
            n=60
            band_center=150
            band_width=20
        field_of_view=0.8
        baseline=0
        geometry=0
        primary_size=6

        start_time=2022-02-10T06:00:00
        integration_time=60 # in seconds
        pointing_frame=az_el
        degrees=True
        sample_rate=20
        scan_pattern=daisy
        scan_center=(10, 4.5)
        scan_options={}

        region='chajnantor'
        latitude=-23.0294
        longitude=-67.7548
        altitude=5064
        site_documentation=''
        weather_quantiles={}

        atmosphere_model='2d'
        min_atmosphere_beam_res=4
        min_atmosphere_height=500
        max_atmosphere_height=5000
        turbulent_outer_scale=800
        pwv_rms_frac=0.03
        pwv=2

        map_file=''
        map_frame=ra_dec
        map_center=[10, 4.5]
        map_res=0.5
        map_inbright:
        map_units=K_RJ
        map_freqs=[150]

        white_noise_level=1.e-2 # in Kelvin Rayleigh-Jeans equivalent
        pink_noise_level=1.e-2 # in Kelvin Rayleigh-Jeans equivalent amplitude in fourier domain
        pink_noise_slope=0.5

            )
