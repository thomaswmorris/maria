import numpy as np
import pandas as pd


class ReadOutBLIPs:
    """
    Background limited estimates for AtLAST based of from
    Sean Bryans estimator.
    See https://github.com/atlast-telescope/sens_calc for more info.
    """

    def _run(self):
        self._get_levels()

    def _get_levels(self):
        df = pd.read_csv(
            "../sens_calc/results_of_instrument_simulation.csv", index_col=0
        )
        frq_index = np.argmin(np.abs(self.f_c - df["fmin"]))

        used_b_w = df["fmax"][frq_index] - df["fmin"][frq_index]  # GHz
        NET = df["NET_RJ of a Single Detector"][frq_index]

        self.white_noise_RJ = NET / np.sqrt(self.b_w / used_b_w)  # uK s^0.5
        self.pink_noise_RJ = 0.3 * (self.white_noise_RJ / 112.0)


class InitNoise(ReadOutBLIPs):
    """
    Estimate initial white and pink noise levels
    """

    def __init__(
        self,
        f_c: np.float32 = 93,  # GHz -- central frequency
        b_w: np.float32 = 52,  # GHz -- band width
        *args,
        **kwargs,
    ):
        super(InitNoise, self).__init__(*args, **kwargs)

        self.f_c = f_c
        self.b_w = b_w
        self._run()
