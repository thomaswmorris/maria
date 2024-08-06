import pytest

import maria


@pytest.mark.parametrize("site_name", maria.all_sites)
def test_get_site(site_name):
    site = maria.get_site(site_name)
    print(site)
