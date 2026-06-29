.. _sites:

#####
Sites
#####

Overview
========

The observing site is represented by a ``Site``. For example:

.. plot:: 
   :include-source: True

    import maria

    cerro_toco = maria.get_site("cerro_toco")
    cerro_toco.plot()

.. hint:: To see all available pre-defined sites, run ``print(maria.all_sites)``.
