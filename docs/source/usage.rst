=====
Usage
=====

++++++
Arrays
++++++

One component of the simulation is the `Array`, which defines the observing instrument. We can load arrays with::

    import maria

    mustang = maria.get_array("MUSTANG-2") # MUSTANG 2

Available arrays are listed with::

    maria.available_arrays

We can specify a custom array::

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
    - max_baseline (float): meters (only for ALMA)


+++++++++
Pointings
+++++++++

One component of the simulation is the `Array`, which defines the observing instrument.

    import maria

    mustang = maria.get_array("MUSTANG-2")

We can specify a custom array


+++++
Sites
+++++

The last component of the simulation is the `Site`, which represents a specific point

    gbo = maria.get_site("GBO") # Green Bank Obervatory

We can change the

    custom_site = maria.get_site(altitude=900, region="green_bank")


The `Site` object should not be confused with the `region`. Each which is used to model meteorological conditions and atmospheric emission spectra.
