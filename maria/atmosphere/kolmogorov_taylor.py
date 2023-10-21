class KolmogorovTaylorSimulation:
    def __init__(
        self,
        time,
        azim,
        elev,
        baseline,
        field_of_view,
        min_frequency,
        wind_velocity,
        wind_direction,
        extrusion_step=1,
        outer_scale=600,
        min_res=10,
        max_res=100,
        max_depth=3000,
    ):
        self.time = time
        self.azim = azim
        self.elev = elev

        self.baseline = baseline
        self.field_of_view = field_of_view
        self.min_frequency = min_frequency
        self.wind_direction = wind_direction
        self.wind_velocity = wind_velocity
        self.extrusion_step = extrusion_step
        self.min_res = min_res
        self.max_res = max_res
        self.max_depth = max_depth

        self.outer_scale = outer_scale

        (self.cX, self.cZ), self.Y, self.cres = get_extrusion_products(
            self.time,
            self.azim,
            self.elev,
            self.baseline,
            self.field_of_view,
            self.min_frequency,
            self.wind_velocity,
            self.wind_direction,
            self.max_depth,
            self.extrusion_step,
            self.min_res,
            self.max_res,
        )

        self.layer_heights, self.layer_index = np.unique(self.cZ, return_inverse=True)
        self.n_layers = len(self.layer_heights)

        self.dY, self.n_cells = np.gradient(self.Y).mean(), len(self.cX)
        self.n_baseline = len(self.baseline)
        self.n_extrusion = len(self.Y)
        self.n_history = int(2 * self.outer_scale / self.dY) + 1

        self.initialized = False

    def initialize(self):
        max_spacing = int(self.n_cells / 1.1)
        iter_samples = [
            np.random.choice(
                self.n_cells,
                int(self.n_cells / np.minimum(2**i, max_spacing)),
                replace=False,
            )
            for i in range(self.n_history)
        ]
        self.hist_iter_index = np.concatenate(
            [i * np.ones(len(index), dtype=int) for i, index in enumerate(iter_samples)]
        )
        self.cell_iter_index = np.concatenate(iter_samples).astype(int)

        Xi, Xj = self.cX, self.cX[self.cell_iter_index]
        Yi, Yj = np.zeros(self.n_cells), self.dY * (1 + self.hist_iter_index)
        Zi, Zj = self.cZ, self.cZ[self.cell_iter_index]

        Rii = np.sqrt(
            np.subtract.outer(Xi, Xi) ** 2
            + np.subtract.outer(Yi, Yi) ** 2
            + np.subtract.outer(Zi, Zi) ** 2
        )

        Rij = np.sqrt(
            np.subtract.outer(Xi, Xj) ** 2
            + np.subtract.outer(Yi, Yj) ** 2
            + np.subtract.outer(Zi, Zj) ** 2
        )

        Rjj = np.sqrt(
            np.subtract.outer(Xj, Xj) ** 2
            + np.subtract.outer(Yj, Yj) ** 2
            + np.subtract.outer(Zj, Zj) ** 2
        )

        alpha = 1e-3

        # this is all very computationally expensive stuff (n^3 is rough!)
        self.Cii = utils.approximate_matern(
            Rii, r0=self.outer_scale, nu=1 / 3, n_test_points=4096
        ) + alpha**2 * np.eye(self.n_cells)
        self.Cij = utils.approximate_matern(
            Rij, r0=self.outer_scale, nu=1 / 3, n_test_points=4096
        )
        self.Cjj = utils.approximate_matern(
            Rjj, r0=self.outer_scale, nu=1 / 3, n_test_points=4096
        )
        self.A = np.matmul(self.Cij, utils.fast_psd_inverse(self.Cjj))
        self.B, _ = sp.linalg.lapack.dpotrf(self.Cii - np.matmul(self.A, self.Cij.T))

        self.initialized = True

    def extrude(self):
        if not self.initialized:
            self.initialize()

        ARH = np.zeros((self.n_cells, self.n_history))
        self.cdata = np.zeros((self.n_cells, self.n_extrusion))

        def iterate_autoregression(ARH, n_iter):
            for i in range(n_iter):
                res = np.matmul(
                    self.A, ARH[self.cell_iter_index, self.hist_iter_index]
                ) + np.matmul(self.B, np.random.standard_normal(self.n_cells))
                ARH = np.c_[res, ARH[:, :-1]]
            return ARH

        ARH = iterate_autoregression(ARH, n_iter=2 * self.n_history)

        for i in range(self.n_extrusion):
            ARH = iterate_autoregression(ARH, n_iter=1)
            self.cdata[:, i] = ARH[:, 0]


def get_extrusion_products(
    time,
    azim,
    elev,
    baseline,
    field_of_view,
    min_frequency,
    wind_velocity,
    wind_direction,
    max_depth=3000,
    extrusion_step=1,
    min_res=10,
    max_res=100,
):
    """
    For the Kolomogorov-Taylor model. It works the same as one of those pasta extruding machines (https://youtu.be/TXtm_eNaIwQ). This function figures out the shape of the hole.

    The inputs are:

    azim, elev: time-ordered pointing
    """

    # Within this function, we work in the frame of wind-relative pointing phi'. The elevation stays the same. We define:
    #
    # phi' -> phi - phi_w
    #
    # where phi_w is the wind bearing, i.e. the direction the wind comes from. This means that:
    #
    # For phi' = 0°  you are looking into the wind
    # For phi' = 90° the wind is going from left to right in your field of view.
    #
    # We further define the frame (x, y, z) = (r cos phi' sin theta, r sin phi' sin theta, r sin theta)

    n_baseline = len(baseline)

    # this converts the $(x, y, z)$ vertices of a beam looking straight up to the $(x, y, z)$ vertices of
    # the time-ordered pointing of the beam in the frame of the wind direction.
    elev_rotation_matrix = sp.spatial.transform.Rotation.from_euler(
        "y", np.pi / 2 - elev
    ).as_matrix()
    azim_rotation_matrix = sp.spatial.transform.Rotation.from_euler(
        "z", np.pi / 2 - azim + wind_direction
    ).as_matrix()
    total_rotation_matrix = np.matmul(azim_rotation_matrix, elev_rotation_matrix)

    # this defines the maximum size of the beam that we have to worry about
    max_beam_radius = _beam_sigma(max_depth, primary_size=primary_sizes, nu=5e9)
    max_boresight_offset = max_beam_radius + max_depth * field_of_view / 2

    # this returns a list of time-ordered vertices, which can used to construct a convex hull
    time_ordered_vertices_list = []
    for _min_radius, _max_radius, _baseline in zip(
        primary_sizes / 2, max_boresight_offset, baseline
    ):
        _rot_baseline = np.matmul(
            sp.spatial.transform.Rotation.from_euler("z", wind_direction).as_matrix(),
            _baseline,
        )

        _vertices = np.c_[
            np.r_[
                [
                    _.ravel()
                    for _ in np.meshgrid(
                        [-_min_radius, _min_radius], [-_min_radius, _min_radius], [0]
                    )
                ]
            ],
            np.r_[
                [
                    _.ravel()
                    for _ in np.meshgrid(
                        [-_max_radius, _max_radius],
                        [-_max_radius, _max_radius],
                        [max_depth],
                    )
                ]
            ],
        ]

        time_ordered_vertices_list.append(
            _rot_baseline[None]
            + np.swapaxes(np.matmul(total_rotation_matrix, _vertices), 1, -1).reshape(
                -1, 3
            )
        )

    # these are the bounds in space that we have to worry about. it has shape (3, 2)
    extrusion_bounds = np.c_[
        np.min([np.min(tov, axis=0) for tov in time_ordered_vertices_list], axis=0),
        np.max([np.max(tov, axis=0) for tov in time_ordered_vertices_list], axis=0),
    ]

    # here we make the layers of the atmosphere given the bounds of $z$ and the specified resolution; we use
    # regular layers to make interpolation more efficient later
    min_height = np.max(
        [
            np.max(tov[:, 2][tov[:, 2] < np.median(tov[:, 2])])
            for tov in time_ordered_vertices_list
        ]
    )
    max_height = np.min(
        [
            np.min(tov[:, 2][tov[:, 2] > np.median(tov[:, 2])])
            for tov in time_ordered_vertices_list
        ]
    )

    height_samples = np.linspace(min_height, max_height, 1024)
    dh = np.gradient(height_samples).mean()
    dheight_dindex = np.interp(
        height_samples, [min_height, max_height], [min_res, max_res]
    )
    dindex_dheight = 1 / dheight_dindex
    n_layers = int(np.sum(dindex_dheight * dh))
    layer_heights = sp.interpolate.interp1d(
        np.cumsum(dindex_dheight * dh),
        height_samples,
        bounds_error=False,
        fill_value="extrapolate",
    )(1 + np.arange(n_layers))

    # here we define the cells through which turbulence will be extruded.
    layer_res = np.gradient(layer_heights)
    x_min, x_max = extrusion_bounds[0, 0], extrusion_bounds[0, 1]
    n_per_layer = ((x_max - x_min) / layer_res).astype(int)
    cell_x = np.concatenate(
        [
            np.linspace(x_min, x_max, n)
            for i, (res, n) in enumerate(zip(layer_res, n_per_layer))
        ]
    )
    cell_z = np.concatenate(
        [h * np.ones(n) for i, (h, n) in enumerate(zip(layer_heights, n_per_layer))]
    )
    cell_res = np.concatenate(
        [res * np.ones(n) for i, (res, n) in enumerate(zip(layer_res, n_per_layer))]
    )

    extrusion_cells = np.c_[cell_x, cell_z]
    extrusion_shift = np.c_[cell_res, np.zeros(len(cell_z))]

    eps = 1e-6

    in_view = np.zeros(len(extrusion_cells)).astype(bool)
    for tov in time_ordered_vertices_list:
        baseline_in_view = np.zeros(len(extrusion_cells)).astype(bool)

        hull = sp.spatial.ConvexHull(
            tov[:, [0, 2]]
        )  # we only care about the $(x, z)$ dimensions here
        A, b = hull.equations[:, :-1], hull.equations[:, -1:]

        for shift_factor in [-1, 0, +1]:
            baseline_in_view |= np.all(
                (extrusion_cells + shift_factor * extrusion_shift) @ A.T + b.T < eps,
                axis=1,
            )

        in_view |= (
            baseline_in_view  # if the cell is in view of any of the baselines, keep it!
        )

        # plt.scatter(tov[:, 0], tov[:, 2], s=1e0)

    # here we define the extrusion axis
    y_min, y_max = (
        extrusion_bounds[1, 0],
        extrusion_bounds[1, 1] + wind_velocity * time.ptp(),
    )
    extrustion_axis = np.arange(y_min, y_max, extrusion_step)

    return extrusion_cells[in_view].T, extrustion_axis, cell_res[in_view]
