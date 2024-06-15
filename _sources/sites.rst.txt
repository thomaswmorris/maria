#####
Sites
#####

The observing site is represented by a ``Site``. For example::

.. code-block:: python

    cerro_toco = maria.get_site("cerro_toco")

+++++++++++++++++
Customizing Sites
+++++++++++++++++

We can customize sites as::

    taller_cerro_toco = maria.get_site("cerro_toco", altitude=6190) # one kilometer higher!
