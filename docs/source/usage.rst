Usage
#####



Objects
=======

Arrays
^^^^^^

One component of the simulation is the ``Array``, which defines the observing instrument. This includes the set of detectors which make observations together, as well as information about the telescope like the aperture size. ``maria`` supports several real-life arrays; we can load e.g. the `MUSTANG-2 <www.nrao.edu>`_ array as::

    import maria

    mustang = maria.get_array("MUSTANG-2")

A database of supported arrays can be found at::

    maria.available_arrays

Supported arrays are customizeable by passing extra keywords to the ``get_array()`` function. We can double the size of the Green Bank Telescope and increase MUSTANG's field of view to one degree with::

    mustang = maria.get_array("MUSTANG-2",
                              field_of_view=10, # in degrees
                              primary_size=200 # in meters
                              )

The arrays

- **array** telescope and array design parameters
    - Predefined configurations: `MUSTANG-2`, `ALMA`, `AtLAST`, `SCUBA-2`
    - specific keys:
        - dets (dict): {'float': [number of detectors (int), band_center in Hz (float), band_width in Hz (float)]}
        - field_of_view (float): in degrees
        - geometry (string): options are `hex` (defeault), `flower`, `square`
        - primary_size (float): meters
        - az_bounds (list): [lower (float), upper (float)] in degrees
        - el_bounds (list): [lower (float), upper (float)] in degrees
        - max_az_vel (float): ...
        - max_el_vel (float): ...
        - max_az_acc (float): ...
        - max_el_acc (float): ...
        - baseline (float): meters (only for ALMA)


Pointings
^^^^^^^^^

The next component of the simulation is the pointing, which defines the direction and scan pattern of the observation:::

    daisy_scan = maria.get_pointing("daisy")

We can specify a custom pointing by passing extra arguments to the above, such as::

    custom_daisy_scan = maria.get_pointing("daisy", integration_time=600, scan_options={"radius": 2})


Sites
^^^^^

The last component of the simulation is the `Site`, which represents a specific point

    gbo = maria.get_site("GBO") # Green Bank Obervatory

We can change the

    custom_site = maria.get_site(altitude=900, region="green_bank")


The `Site` object should not be confused with the `region`. Each which is used to model meteorological conditions and atmospheric emission spectra.


Running simulations
===================

Generating time-ordered data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this section, we provide a brief explanation of how to create synthetic time streams using Maria. For a comprehensive overview, please refer to the Tutorials.

The following few lines show how to generate the time streams in Python::

    from maria import Simulation

    sim = Simulation(array='MUSTANG-2',
                     pointing='daisy',
                     site='GBT',
                     atm_model='single_layer',
                     map_file=some_input_file)



Running the simulation returns a ``TOD`` object::

    tod = sim.run()

The time-ordered data can be accessed as ``tod.data``, and consists of a timestream per detector. The ``Simulation`` class accepts many keywords that modify the behavior of the simulation are outlined below.


Simulation parameters
^^^^^^^^^^^^^^^^^^^^^

- **array** telescope and array design parameters
    - Predefined configurations: `MUSTANG-2`, `ALMA`, `AtLAST`, `SCUBA-2`
    - specific keys:
        - dets (Mappable of Mappables): {'float': [number of detectors (int), band_center in Hz (float), band_width in Hz (float)]}
        - field_of_view (float): in degrees
        - geometry (string): options are `hex` (defeault), `flower`, `square`
        - primary_size (float): meters
        - az_bounds (list): [lower (float), upper (float)] in degrees
        - el_bounds (list): [lower (float), upper (float)] in degrees
        - max_az_vel (float): ...
        - max_el_vel (float): ...
        - max_az_acc (float): ...
        - max_el_acc (float): ...
        - baseline (float): meters (only for ALMA)

- **pointing:** Scanning strategy
    - Predefined configurations: `stare`, `daisy`, `BAF`,
    - specific keys:
        - start_time (string): reference point for generating weather, example: '2022-02-10T06:00:00'
        - integration_time (float): in seconds
        - scan_pattern (string):  options are `daisy` or `back-and-forth`
        - pointing_frame (string): options are `az_el` or `ra_dec`
        - scan_center (list): [RA (float), Dec (float)] in degree
        - scan_radius (float): in meters
        - scan_period (float): in seconds
        - scan_rate (float): in seconds

- **site:** Site locations
    - Predefined configurations: `APEX`, `ACT`, `GBT`, `JCMT`, `SPT`, `SRT`
    - specific keys:
        - region (string): options are `chajnantor`, `green_bank`, `mauna_kea`, `south_pole`, `sardinia`
        - latitude (float): in degree
        - longtitude (float): in degree
        - altitude (float): in meters
        - seasonal (bool):
        - diurnal (bool):
        - weather_quantiles (dict): keys: `column_water_vapor` (float),  ...
        - pwv_rms (float): ...

- **atm_model:** Different atmospheric models
    - Predefined configurations: `single_layer`, None
    - specific keys:
        - min_depth (float): in meters
        - max_depth (float): in meters
        - n_layers (int): number of atmospheric layers
        - min_beam_res (int):

- **mapper:** Different mappers
    - Only one mapper is implemented, the `BinMapper`
    - specific keys:
        - map_height (float): radians
        - map_width (float): radians
        - map_res (float): radians
        - map_filter (bool): Fourier filter the time streams before common-mode subtraction
        - n_modes_to_remove (int): number of eigen modes to remove. Set to 0 for no common-mode subtraction.

- **sky:** Input file
    - specific keys:
        - map_file (string): `path_to_fits_file.fits`
        - map_frame (string): options are `az_el` or `ra_dec`
        - map_center (list): [RA (float), Dec (float)] in degree
        - map_res (float): in degrees
        - map_inbright (float): scale the map so the brightest pixel value becomes this value
        - map_units (string): options are `K_RJ` or `Jy/pixel`



Mapping time-ordered data
^^^^^^^^^^^^^^^^^^^^^^^^^

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
