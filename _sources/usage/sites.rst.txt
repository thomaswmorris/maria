#####
Sites
#####

The observing site is represented by a ``Site``. For example:

.. code-block:: python

    cerro_toco = maria.get_site("cerro_toco")

To see the list of supported sites, run

.. code-block:: python

    print(maria.all_sites)

+++++++++++++++++
Customizing Sites
+++++++++++++++++

We can customize sites as:

.. code-block:: python

    taller_cerro_toco = maria.get_site("cerro_toco", altitude=6190) # one kilometer higher!
