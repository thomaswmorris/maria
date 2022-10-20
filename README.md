# Modular Auto-Regressive Integrated Atmosphere (maria) 
[![name](https://img.shields.io/pypi/v/maria.svg)](https://pypi.python.org/pypi/maria) [![name](https://img.shields.io/travis/tomachito/maria.svg)](https://travis-ci.org/tomachito/maria)

[Oh, maria blows the stars around / and sends the clouds a-flying](https://youtu.be/qKxgfnoz2pk)

maria is a python-based package that simulates turbulent atmospheric emission using a auto-regressive Gaussian process framework, for applications in observational astronomy. Below: a distribution of turbulent water vapor moves through the field of view of the observer. 

![Watch the video](https://user-images.githubusercontent.com/41275226/117068746-acbf8980-acf9-11eb-8016-64fa01e12a77.mp4)

## Background

Atmospheric modeling is an important step in both experiment design and subsequent data analysis for ground-based cosmological telescopes observing the cosmic microwave background (CMB). The next generation of ground-based CMB experiments will be marked by a huge increase in data acquisition: telescopes like [AtLAST](https://www.atlast.uio.no) and [CMB-S4](https://cmb-s4.org) will consist of hundreds of thousands of superconducting polarization-sensitive bolometers sampling the sky. This necessitates new methods of efficiently modeling and simulating atmospheric emission at small angular resolutions, with algorithms than can keep up with the high throughput of modern telescopes.

maria simulates layers of turbulent atmospheric emission according to a statistical model derived from observations of the atmosphere in the Atacama Desert, from the [Atacama Cosmology Telescope (ACT)](https://lambda.gsfc.nasa.gov/product/act/) and the [Atacama B-Mode Search (ABS)](https://lambda.gsfc.nasa.gov/product/abs/). It uses a sparse-precision auto-regressive Gaussian process algorithm that allows for both fast simulation of high-resolution atmosphere, as well as the ability to simulate arbitrarily long periods of atmospheric evolution. 

## Methodology

maria auto-regressively simulates an multi-layeed two-dimensional "integrated" atmospheric model that is much cheaper to compute than a three-dimensional model, which can effectively describe time-evolving atmospheric emission. maria can thus effectively simulate correlated atmospheric emission for in excess of 100,000 detectors observing the sky concurrently, at resolutions as fine as one arcminute. The atmospheric model used is detailed [here](https://arxiv.org/abs/2111.01319).

## Examples and Usage 

To install MARIA with PyPi, run

```console
pip install maria
```
The main tool of the maria module is the model object. The default model can be easily intitialized as 

```python
import maria

default_model = maria.model()
```

Different models can be initialized by configurating different aspects of the model.

### Arrays

The array config defines the set of detectors that observe the sky, along with the properties of their optics and noise. The array is specified as a dictionary

```python                       
array_config = {'shape' : 'hex',   # The shape of the distribution of detectors. Supported shapes are `hex', 'square', and 'flower'. 
                    'n' : 10000,   # The number of detectors in the array.  
                  'fov' : 2,       # The maximum width of the array's field-of-view on the sky, in degrees. 
                 'band' : 1.5e11}  # The observing band of the detector, in Hz. 
```
Alternatively, the array can be configured manually by supplying an array of values for each parameter. In this case, the first three parameters are replaced by

```python
array_config={'offset_x' : some_array_of_offsets_x,  # in degrees
              'offset_y' : some_array_of_offsets_y}) # in degrees
```

### Observations

The pointing config defines the time-ordered parameters of the simulation. Below is the config for a constant-elevation scan (CES) that observes at 90+/-45 degrees of azimuth and 60 degrees of elevation, sampling at 100Hz for ten minutes. 

```python
pointing_config = {'scan_type' : 'CES', # scan pattern
                    'duration' : 600,   # duration of the observation, in seconds 
                   'samp_freq' : 100,   # sampling rate, in Hz
                 'center_azim' : 90,    # azimuth of the center of the scan, in degrees
                    'az_throw' : 45,    # half of the azimuth width of the scan, in degrees
                 'center_elev' : 60,    # observing elevation of the scan, in degrees
                    'az_speed' : 1.5}   # scanning speed of the array, in degrees per second
```
Alternatively, the pointing data may be given manually
```python
pointing_config = {'time' : some_array_of_timestamps, # in seconds
             'focal_azim' : some_array_of_azimuths,   # in degrees
             'focal_elev' : some_array_of_elevations} # in degrees
```
where focal_azim and focal_elev describe the angular pointing of the center of the array. 

### Sites

The site determines the motion of celestial sources as the earth rotates under the telescope, as well as the observing conditions. Below is the config for the ACT site:
```python
site_config = {'site' : 'ACT', 
               'time' : datetime.now(timezone.utc).timestamp(),
 'weather_gen_method' : 'random'} 
```
Weather data are quantitatively realistic for a given site, altitude, time of day, and time of year, and are generated using the [weathergen](https://github.com/tomachito/weathergen) package. 

### Models

The model defines the 

```python
atmosphere_config = {'n_layers'        : 10,         # how many layers to simulate, based on the integrated atmospheric model 
                    'min_depth'        : 500,        # the distance of the first layer from the telescope, in meters
                    'max_depth'        : 10000,      # the distance of the second layer, in meters
                    'atmosphere_rms'   : 50,         # the total RMS of atmospheric noise, in mK_CMB
                    'outer_scale'      : 500}        # the outer scale of spatial fluctuations in emission, in meters
```

## Examples

Passing these dictionaries as arguments produces a customized model

```python
import maria

my_model = model(atmosphere_config=atmosphere_config,
                 pointing_config=pointing_config,
                 beams_config=beams_config,
                 array_config=array_config,
                 site_config=site_config)
```
Data can then be simulated from the model by running 

```python
data = my_model.sim()
```
which returns an array where the first dimension corresponds to detector index, and the second dimension to the timesample index. 

### Caution

Gaussian process regression has cubic complexity, which scales poorly (especially when coded in Python). Simulating large swaths of atmosphere at high resolutions can be extremely slow, so don't go crazy with the input parameters. 

This package also produces large arrays: 1000 detectors sampling at 50 Hz for an hour is well over a gigabyte of data.

