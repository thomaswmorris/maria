.. _plans:

#####
Plans
#####

Overview
--------

How an instrument observes is represented as a ``Plan``. Because it can be difficult to tell when a given source will be far enough above the horizon to observe, the easiest way to generate a plan is with a ``Planner``.

++++++++
Planners
++++++++

We can generate plans with the ``Planner`` as

.. plot:: 
   :include-source: True

    import maria
    from maria import Planner

    input_map = maria.map.load(maria.io.fetch("maps/crab_nebula.fits")) # load an example map

    planner = Planner(target=input_map, 
                      site="green_bank", 
                      constraints={"el": (70, 90), # in degrees
                                   "min_sun_distance": 20, # in degrees
                                   "hour": (14, 15)})

    plan = planner.generate_plan(total_duration=600, # in seconds
                                 scan_options={"radius": input_map.width.deg / 3}, # in degrees
                                 sample_rate=50) # in Hz

    plan.plot()


Passing parameters to ``generate_plan`` will be passed to the ``Plan``.

+++++++++++++++++
Customizing Plans
+++++++++++++++++

A 60-second zenith stare would be instantiated as

.. code-block:: python

    stare = maria.get_plan(start_time="2022-02-10T06:00:00",
                           scan_pattern="stare"
                           duration=60, # in seconds
                           sample_rate=20, # in Hz
                           pointing_frame="az/el",
                           scan_center=(0, 90)) # in degrees

We might also do a daisy scan on some given point on the sky, which the telescope will track:

.. code-block:: python

    tracking_daisy = maria.get_plan(start_time="2022-02-10T06:00:00",
                                    scan_pattern="daisy",
                                    scan_options={"radius": 0.5, "speed": 0.1}, # in degrees
                                    duration=600, # in seconds
                                    sample_rate=50, # in Hz
                                    frame="ra/dec"
                                    scan_center=(150, 10)) # in degrees
