#####
Plans
#####

The observing plan is represented by a ``Plan``. A 60-second zenith stare would be instantiated as

.. code-block:: python

    stare = maria.get_plan(start_time='2022-02-10T06:00:00',
                           duration=60,
                           pointing_frame='az_el',
                           scan_center=(0, 90),
                           scan_pattern='stare')
