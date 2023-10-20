Code Documentation
==================

In this documentation, you can find an overview of available keywords for customizing mock observations using the Maria software. We've included predefined configurations for convenience, which set all keywords to match specific settings.
Below we list the keywords which compose the respective catagories: 

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
    - map_units (string): options are `KRJ` or `Jy/pixel`