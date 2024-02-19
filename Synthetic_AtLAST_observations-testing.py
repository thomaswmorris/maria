import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from maria import Simulation
from maria.map.mappers import BinMapper

matplotlib.rc("image", origin="lower")
CMAP = "RdBu_r"


# To make the scan evolve over night we need to update the obs time
# below is a wrapper
# ------------------
def time_to_secs(time):
    _ = time.split(":")
    return float(_[0]) * 60 * 60 + float(_[1]) * 60 + float(_[2])


def secs_to_time(secs):
    hour = secs // (60 * 60)
    mins = secs % (60 * 60) // 60
    secs = secs % (60)
    return hour, mins, secs


def update_time(begin_time, integration_time):
    day = begin_time.split("T")[0]
    time = begin_time.split("T")[1]

    seconds = time_to_secs(time)
    seconds += integration_time
    h, m, s = secs_to_time(seconds)

    x_day = h // 24

    new_day = day
    if x_day > 0:
        new_day = "{}-{:02d}-{}".format(
            new_day.split("-")[0],
            int(int(new_day.split("-")[1]) + x_day),
            new_day.split("-")[2],
        )

    new_time = new_day + "T" + f"{int(h%24):02d}:{int(m):02d}:{int(s):02d}"
    return new_time


def run_sim(ndets, freq, band_width, nscans, vis=True, noisy=True):
    # Further et up telescope and instrument
    # -------------
    BEGIN_TIME = "2022-08-01T23:00:00"
    scan_time = 8.6 * 60.0  # seconds

    if noisy:
        atm_model = "2d"
        white_noise = 266e-6 / np.sqrt(52 / 37.4) * 0.3 / 1.113
        pink_noise = 0.6 * 0.3 / 1.113
    else:
        atm_model = None
        white_noise = 0.0
        pink_noise = 0.0

    pointing_center = (260.0, -10.0)  # RA and Dec in degrees
    pixel_size = 0.0009765625  # degree

    scanning_speed = 0.5  # deg/s
    sample_rate = float(int(225.0 * (scanning_speed / 0.5) * (freq / 92.0) ** 2))  # Hz

    field_of_view = 1.0  # deg
    scanning_radius = field_of_view / 2
    ndets_2f = float(int(3000 * (field_of_view / 0.25) ** 2 * (freq / 92.0) ** 2))

    print(f"Should use {int(ndets_2f)} detectors, we use {ndets} dets")
    print(f"Sample rate is {int(sample_rate)} Hz")

    # Initialize mapper
    fov = (field_of_view * u.degree).to(u.arcsec)
    scn_velocity = scanning_speed * u.degree / u.s
    filter_freq = (scn_velocity / fov).to(u.Hz).value / 2

    mapper = BinMapper(
        center=(np.radians(pointing_center[0]), np.radians(pointing_center[1])),
        frame="ra_dec",
        width=np.radians(field_of_view + scanning_radius * 3),
        height=np.radians(field_of_view + scanning_radius * 3),
        res=np.radians(4.0 / 3600.0),
        degrees=False,
        calibrate=True,
        tod_postprocessing={
            "remove_modes": {"n": 1},
            "highpass": {"f": filter_freq},
            "despline": {},
        },
        map_postprocessing={"gaussian_filter": {"sigma": 2}},
    )

    for i in range(nscans):
        print(f"Scan number {i}, taken at: {update_time(BEGIN_TIME, scan_time*i)}")
        sim = Simulation(
            # Mandatory weather settings
            # ---------------------
            instrument="AtLAST",
            pointing="daisy",
            site="llano_de_chajnantor",
            # Noise settings:
            # ----------------
            atmosphere_model=atm_model,
            pwv_rms_frac=0.05,
            # True sky input
            # ---------------------
            map_file=inputfile,
            map_units="Jy/pixel",
            map_res=pixel_size,
            map_center=pointing_center,
            map_freqs=[FREQ],
            # AtLAST Observational setup
            # ----------------------------
            integration_time=scan_time,
            sample_rate=sample_rate,
            scan_center=pointing_center,
            start_time=update_time(BEGIN_TIME, scan_time * i),
            pointing_frame="ra_dec",
            field_of_view=field_of_view,
            bands={
                "f090": {
                    "n_dets": ndets,
                    "band_center": freq,
                    "band_width": band_width,
                    "white_noise": white_noise,
                    "pink_noise": pink_noise,
                }
            },
            scan_options={
                "speed": scanning_speed,
                "radius": scanning_radius,
                "petals": 2.11,
            },
            verbose=vis,
        )

        tod = sim.run()
        mapper.add_tods(tod)

    mapper.run()
    mapper.save_maps(outfile_map)

    if vis:
        # - Input figure
        hdu = fits.open(inputfile)
        header = hdu[1].header
        wcs_input = WCS(header, naxis=2)

        # scanning of one tod
        fig = plt.figure(dpi=256, tight_layout=True)
        fig.set_size_inches(12, 5, forward=True)
        fig.suptitle("Scanning strategy")

        # - Plot
        ax = plt.subplot(1, 2, 1)

        ax.plot(np.degrees(tod.boresight.az), np.degrees(tod.boresight.el), lw=5e-1)
        ax.set_xlabel("az (deg)"), ax.set_ylabel("el (deg)")

        ax = plt.subplot(1, 2, 2, projection=wcs_input)
        im = ax.imshow(hdu[1].data, cmap=CMAP)

        ra, dec = ax.coords
        ra.set_major_formatter("hh:mm:ss")
        dec.set_major_formatter("dd:mm:ss")
        ra.set_axislabel(r"RA [J2000]", size=11)
        dec.set_axislabel(r"Dec [J2000]", size=11)
        ra.set_separator(("h", "m"))

        sky = SkyCoord(
            np.degrees(tod.boresight.ra) * u.deg, np.degrees(tod.boresight.dec) * u.deg
        )
        pixel_sky = wcs_input.world_to_pixel(sky)
        ax.plot(pixel_sky[0], pixel_sky[1], lw=5e-1, alpha=0.5)
        ax.set_xlabel("ra (deg)"), ax.set_ylabel("dec (deg)")
        plt.savefig(outfile_map.replace(".fits", "_single_scan.pdf"), dpi=fig.dpi)
        plt.close()

        # - Mapper scan and hit map:
        # --------
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(10, 4.5),
            dpi=256,
            tight_layout=True,
            gridspec_kw={"width_ratios": [1, 1.5]},
        )
        fig.suptitle("Detector setup for one band")

        for uband in sim.instrument.bands:
            axes[0].plot(
                60 * np.degrees(tod.boresight.ra - tod.boresight.ra.mean()),
                60 * np.degrees(tod.boresight.dec - tod.boresight.dec.mean()),
                lw=5e-1,
                alpha=0.5,
                label="Scanning Pattern",
            )
            axes[0].scatter(
                60 * np.degrees(sim.instrument.offset_x),
                60 * np.degrees(sim.instrument.offset_y),
                label=f"{uband} mean",
                lw=5e-1,
                c="orange",
            )

        axes[0].set_xlabel(r"$\theta_x$ offset (arcminutes)")
        axes[0].set_ylabel(r"$\theta_y$ offset (arcminutes)")

        xs, ys = np.meshgrid(
            60 * np.rad2deg((mapper.x_bins[1:] + mapper.x_bins[:-1]) / 2),
            60 * np.rad2deg((mapper.y_bins[1:] + mapper.y_bins[:-1]) / 2),
        )

        im = axes[1].pcolormesh(
            xs,
            ys,
            mapper.raw_map_cnts[tod.dets.band[0]],
            label="Photon counts in band " + tod.dets.band[0],
            rasterized=True,
        )

        axes[1].set_xlabel(r"$\theta_x$ (arcmin)"), axes[1].set_ylabel(
            r"$\theta_y$ (arcmin)"
        )
        cbar = plt.colorbar(im, ax=axes[1])
        cbar.set_label("Counts")
        plt.savefig(outfile_map.replace(".fits", "_hitmap.pdf"), dpi=fig.dpi)
        plt.close()

        # - Plot Mock observation
        # ---------------
        outputfile = outfile_map

        hdu_out = fits.open(outputfile)
        wcs_output = WCS(hdu_out[0].header, naxis=2)

        fig = plt.figure(dpi=512, tight_layout=False)
        fig.set_size_inches(6, 4, forward=True)

        ax = plt.subplot(1, 1, 1, projection=wcs_output)

        im = ax.imshow(
            hdu_out[0].data[0] * 1e6,
            cmap=CMAP,
            vmin=-4 * np.nanstd(hdu_out[0].data[0]) * 1e6,
            vmax=4 * np.nanstd(hdu_out[0].data[0]) * 1e6,
            rasterized=True,
        )

        cbar = plt.colorbar(im, ax=ax, shrink=1.0)
        cbar.set_label(r"S$_{\nu}$ [$\mu$Jy/pixel]")

        ra, dec = ax.coords
        ra.set_major_formatter("hh:mm:ss")
        dec.set_major_formatter("dd:mm:ss")
        ra.set_axislabel(r"RA [J2000]", size=11)
        dec.set_axislabel(r"Dec [J2000]", size=11)
        ra.set_separator(("h", "m"))

        ax.invert_xaxis()

        plt.tight_layout()
        plt.savefig(outfile_map.replace(".fits", "_map.pdf"), dpi=fig.dpi)
        plt.close()


if __name__ == "__main__":
    # stuff you want to change.
    # -------------
    NDETS = 300
    FREQ = 92.0  # GHZ
    BAND_WIDTH = 52.0
    NSCANS = 5

    VIS = True
    NOISY = True

    # Load in celestial sky"
    inputfile = "./maps/big_cluster2.fits"
    outfile_map = "./output/FREQ{}_Noisy{}_{}".format(
        int(FREQ), str(NOISY), inputfile.split("/")[-1].replace(".fits", "_out.fits")
    )

    run_sim(NDETS, FREQ, BAND_WIDTH, NSCANS, VIS, NOISY)
