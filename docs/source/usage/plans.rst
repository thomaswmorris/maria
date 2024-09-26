#####
Plans
#####

The observing plan is represented by a ``Plan``. To see the list of supported plans, run

.. code-block:: python

    print(maria.all_instruments)


+++++++++++++++++
Customizing Plans
+++++++++++++++++

A 60-second zenith stare would be instantiated as

.. code-block:: python

    stare = maria.get_plan(start_time="2022-02-10T06:00:00",
                           scan_pattern="stare"
                           duration=60, # in seconds
                           sample_rate=20, # in Hz
                           pointing_frame="az_el",
                           scan_center=(0, 90)) # in degrees



We might also do a daisy scan on some given point on the sky, which the telescope will track:

.. code-block:: python

    tracking_daisy = maria.get_plan(start_time="2022-02-10T06:00:00",
                                    scan_pattern="daisy",
                                    scan_options={"radius": 0.5, "speed": 0.1}, # in degrees
                                    duration=600, # in seconds
                                    sample_rate=50, # in Hz
                                    frame="ra_dec"
                                    scan_center=(150, 10)) # in degrees
