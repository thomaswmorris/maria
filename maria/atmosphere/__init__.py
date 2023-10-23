from .single_layer import SingleLayerSimulation  # noqa F401

# how do we do the bands? this is a great question.
# because all practical telescope instrumentation assume a constant band

# from .kolmogorov_taylor import KolmogorovTaylorSimulation
# from .linear_angular import LinearAngularSimulation

ATMOSPHERE_PARAMS = {
    "min_depth": 500,
    "max_depth": 3000,
    "n_layers": 4,
    "min_beam_res": 4,
}
