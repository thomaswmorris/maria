Sites
=====

The last component of the simulation is the `Site`, which represents a specific point 

    gbo = maria.get_site("GBO") # Green Bank Obervatory

We can change the 

    custom_site = maria.get_site(altitude=900, region="green_bank")


The `Site` object should not be confused with the `region`. Each which is used to model meteorological conditions and atmospheric emission spectra. 