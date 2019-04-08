import numpy as np


def get_wavelength(hdr) -> np.ndarray:
    w0 = hdr['CRVAL1']
    dw = hdr['CDELT1']
    N = hdr['NAXIS1']
    w1 = w0 + N*dw
    return np.linspace(w0, w1, N, endpoint=True)