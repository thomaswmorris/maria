import logging

import numpy as np
import pandas as pd

from maria.utils import compute_diameter, get_rotation_matrix_2d

logger = logging.getLogger("maria")

SHAPES = ["triangle", "square", "hexagon", "octagon", "circle", "rhombus"]
PACKINGS = ["triangular", "square", "sunflower"]


def generate_sunflower_packing(n: int):
    i = np.arange(n)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    x = 0.5966 * np.sqrt(i) * np.cos(golden_angle * i)
    y = 0.5966 * np.sqrt(i) * np.sin(golden_angle * i)

    return pd.DataFrame({"x": x, "y": y})


def generate_square_packing(n_row: int, n_col: int):
    x_side = np.arange(n_col, dtype=float)
    y_side = np.arange(n_row, dtype=float)
    col, row = np.meshgrid(x_side, y_side)

    x = col - n_col // 2 + (n_col + 1) % 2
    y = row - n_row // 2 + (n_row + 1) % 2

    df = pd.DataFrame({"x": x.ravel(), "y": y.ravel(), "row": row.ravel(), "col": col.ravel()})  # noqa
    df = df.sort_values(["row", "col"], ascending=[False, True])  # noqa
    df.index = np.arange(len(df))

    return df


def generate_triangular_packing(n_col: int, n_row: int):
    x_side = np.arange(n_col, dtype=float)
    y_side = np.arange(n_row, dtype=float)
    col, row = np.meshgrid(x_side, y_side)

    x = col - n_col // 2 + (n_col + 1) % 2
    y = row - n_row // 2 + (n_row + 1) % 2 - 0.5 * x
    x *= np.sqrt(3) / 2

    df = pd.DataFrame({"x": x.ravel(), "y": y.ravel(), "row": row.ravel(), "col": col.ravel()})  # noqa
    df = df.sort_values(["row", "col"], ascending=[False, True])  # noqa
    df.index = np.arange(len(df))

    return df


def scaled_distance(x: float, y: float, shape: str, height_scale: float = 1.0):
    NGONS = {"triangle": 3, "square": 4, "hexagon": 6, "octagon": 8, "circle": 1024}

    r = np.sqrt(x**2 + (y / height_scale) ** 2)
    p = np.arctan2(y / height_scale, x)

    if shape in NGONS:
        n_sides = NGONS[shape]
        d = r * np.cos(np.arcsin(np.sin(n_sides / 2 * p)) * 2 / n_sides)

    elif shape == "rhombus":
        d = r * (np.abs(np.cos(p)) / np.sqrt(3) + np.abs(np.sin(p)))

    else:
        raise ValueError()

    return d + 1e-3 * r.max() * p


def generate_2d_pattern(
    n: int = None,
    n_col: int = None,
    n_row: int = None,
    max_diameter: float = None,
    spacing: float = None,
    shape: str = "hexagon",
    rotation: float = 0,
    packing: str = "triangular",
    height_scale: float = 1.0,
    max_iterations: int = 16,
    tol: float = 1e-2,
):
    """
    Generates an [x,y] array of 2D points to serve as an array.
    By convention, the generated points have a nearest-neighbor distance of 1.
    """

    if packing not in PACKINGS:
        raise ValueError(f"Supported array packings are {PACKINGS}.")

    if shape not in SHAPES:
        raise ValueError(f"Supported array shapes are {SHAPES}.")

    # if no detector is supplied, we'll use an iterative method to find the right number.
    # a good first guess might be n = (max_diameter / spacing) ** 2
    n_explicit = (n is not None) or ((n_col is not None) and (n_row is not None))
    n_args = sum([n_explicit, spacing is not None, max_diameter is not None])

    if n_args < 2:
        raise ValueError()

    if not n_explicit:
        iteration = 0
        current_max_diameter = max_diameter / 1e6
        current_n = 3

        # if only the field of view is supplied
        while (iteration < max_iterations) and (np.abs(np.log(current_max_diameter / max_diameter)) > tol):
            offsets = generate_2d_pattern(
                n=current_n,
                spacing=spacing,
                shape=shape,
                rotation=rotation,
                packing=packing,
            )

            current_max_diameter = compute_diameter(offsets)

            logger.debug(
                f"Array generation iteration {iteration + 1} / {max_iterations} (n={current_n}, "
                f"current_diameter={current_max_diameter:.03e}), target_diameter={max_diameter:.03e})."
            )

            n_adjust_factor = max(1e-2, min((max_diameter / current_max_diameter) ** 2, 1e2))
            current_n = int(np.maximum(3, current_n * n_adjust_factor))

            if current_n > 1e6:
                raise RuntimeError()

            iteration += 1

            # print(iteration, current_n, current_max_diameter)

        return offsets

    else:
        if (n is not None) ^ ((n_col is not None) and (n_row is not None)):
            if n is not None:
                n_col = int(2 * np.sqrt(n))
                n_row = int(2 * np.sqrt(n))
        else:
            raise ValueError()

        if packing == "square":
            df = generate_square_packing(n_col=n_col, n_row=n_row)

        elif packing == "triangular":
            df = generate_triangular_packing(n_col=n_col, n_row=n_row)

        elif packing == "sunflower":
            df = generate_sunflower_packing(n=max(n_col, n_row) ** 2)

        if n is None:
            x = 2 * np.abs(df.x) - 0.25
            y = 2 * np.abs(df.y) - 0.25
            subset_index = np.where((x <= n_col) & (y < n_row))[0]

        if n is not None:
            loss = scaled_distance(x=df.x.values, y=df.y.values, shape=shape, height_scale=height_scale)
            subset_index = sorted(np.argsort(loss)[:n])
            df = df.iloc[subset_index]

        X = (get_rotation_matrix_2d(rotation) @ np.stack([df.x.values, df.y.values])).T

        if max_diameter:
            return max_diameter * X / compute_diameter(X)
        return spacing * X
