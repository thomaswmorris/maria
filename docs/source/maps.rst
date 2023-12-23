Mapping
+++++++

Mapping:::

    from maria.mappers import BinMapper

    mapper = BinMapper(center=center_degrees,
                   frame="ra_dec",
                   width=8/60,
                   height=8/60,
                   res=2/3600,
                   degrees=True,
                   filter_tods=False,
                   n_modes_to_remove=1,
                  )
