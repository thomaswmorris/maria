maria
=====

.. image:: https://github.com/thomaswmorris/maria/actions/workflows/testing.yml/badge.svg
   :target: https://github.com/thomaswmorris/maria/actions/workflows/testing.yml

.. image:: https://img.shields.io/pypi/v/maria.svg
   :target: https://pypi.python.org/pypi/maria

.. image:: ./docs/source/_static/cloud.gif
   :width: 256px
   :alt: StreamPlayer

*maria blows the stars around / and sends the clouds a-flyin’*

``maria`` is a complete simulator of ground-based millimeter- and submillimeter-wave telescopes. Tutorials for installation and usage can be found in the `documentation <https://www.thomaswmorris.com/maria>`_.

Background
----------

Atmospheric modeling is an important step in both experiment design and
subsequent data analysis for ground-based cosmological telescopes
observing the cosmic microwave background (CMB). The next generation of
ground-based CMB experiments will be marked by a huge increase in data
acquisition: telescopes like `AtLAST <https://www.atlast.uio.no>`_ and
`CMB-S4 <https://cmb-s4.org>`_ will consist of hundreds of thousands of
superconducting polarization-sensitive bolometers sampling the sky. This
necessitates new methods of efficiently modeling and simulating
atmospheric emission at small angular resolutions, with algorithms than
can keep up with the high throughput of modern telescopes.

maria simulates layers of turbulent atmospheric emission according to a
statistical model derived from observations of the atmosphere in the
Atacama Desert, from the `Atacama Cosmology Telescope
(ACT) <https://lambda.gsfc.nasa.gov/product/act/>`_ and the `Atacama
B-Mode Search (ABS) <https://lambda.gsfc.nasa.gov/product/abs/>`_. It
uses a sparse-precision auto-regressive Gaussian process algorithm that
allows for both fast simulation of high-resolution atmosphere, as well
as the ability to simulate arbitrarily long periods of atmospheric
evolution.

Methodology
-----------

``maria`` auto-regressively simulates an multi-layeed two-dimensional
“integrated” atmospheric model that is much cheaper to compute than a
three-dimensional model, which can effectively describe time-evolving
atmospheric emission. maria can thus effectively simulate correlated
atmospheric emission for in excess of 100,000 detectors observing the
sky concurrently, at resolutions as fine as one arcminute. The
atmospheric model used is detailed
`here <https://arxiv.org/abs/2111.01319>`_.
