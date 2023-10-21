import pytest

import maria


def test_arrays():
    for array_name in maria.array.ARRAY_CONFIGS.keys():
        print(f"getting array {array_name}")
        array = maria.get_array(array_name)
        print(array)


def test_pointings():
    for pointing_name in maria.pointing.POINTING_CONFIGS.keys():
        print(f"getting pointing {pointing_name}")
        pointing = maria.get_pointing(pointing_name)
        print(pointing)


def test_sites():
    for site_name in maria.site.SITE_CONFIGS.keys():
        print(f"getting site {site_name}")
        site = maria.get_site(site_name)
        print(site)
