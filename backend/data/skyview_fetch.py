# when a user types a galaxy name, fetch real astronomican imagery using nasa's
# sky view virtual telescope api

import requests 
from astropy.io import fits # Flexible Image Transport System = an image + metadata
import numpy as np
import io

def fetch_galaxy(name: str, survey="DSS2 Red") -> np.ndarray: 
# type hints: the input must be string, default value of survey is DSS2 Red,
# the function should output array

    url = "https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl" # pointing to skyview api
    params = {
        "Position": name, # MS1 or NGC 1300 (which galaxy)
        "Survey": survey, # (which telescope data)
        "Return": "FITS", # specific format
        "Pixels": "224, 224",
        "Size": "0.1, 0.1" # (how zoomed in)
    }
    eps = 1e-8
    r = requests.get(url, params=params)
    # FITS -> array -> normalize
    with fits.open(io.BytesIO(r.content)) as hdul:
        data = hdul[0].data.astype(np.float32)
    data = (data - data.min()) / (data.max() - data.min() + eps)
    return data