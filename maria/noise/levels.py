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
            "./maria/noise/results_of_instrument_simulation.csv", index_col=0
        )
        frq_index = np.argmin(np.abs(self.f_c - df["fmin"]))

        used_b_w = df["fmax"][frq_index] - df["fmin"][frq_index]  # GHz
        NET = df["NET_RJ of a Single Detector"][frq_index]

        self.white_noise_RJ = NET / np.sqrt(self.b_w / used_b_w) * 1e-6  # K s^0.5
        self.pink_noise_RJ = 0.3 * (self.white_noise_RJ / 112.0e-6)

        if self.cal:
            self.white_noise_RJ = self.white_noise_RJ * 0.3 / self.abscal
            self.pink_noise_RJ = self.pink_noise_RJ * 0.3 / self.abscal


class InitNoise(ReadOutBLIPs):
    """
    Estimate initial white and pink noise levels
    """

    def __init__(
        self,
        f_c: np.float32 = 93,  # GHz -- central frequency
        b_w: np.float32 = 52,  # GHz -- band width
        abscal: np.float32 = 1.0,  # Calibration term due to atmospheric absorption
        cal: bool = True,
        *args,
        **kwargs,
    ):
        super(InitNoise, self).__init__(*args, **kwargs)

        self.f_c = f_c
        self.b_w = b_w
        self.cal = cal
        self.abscal = abscal
        self._run()
