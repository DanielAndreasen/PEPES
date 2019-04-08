from typing import List, Union

import numpy as np
from spectrum_overload import Spectrum as spc

number_type = Union[int, float]
data_type = Union[np.ndarray, List[number_type]]
interval_type = List[number_type]
intervals_type = List[List[number_type]]


class SpectrumAbstraction:
    def __init__(self, wavelength: data_type, flux: data_type, *args, **kwargs) -> None:
        self.wavelength = wavelength
        self.flux = flux
        self.spectrum = spc(xaxis=wavelength, flux=flux)

    @classmethod
    def from_spectrum(cls, spectrum: spc):
        return cls(spectrum.xaxis, spectrum.flux)

    @classmethod
    def from_file(cls, fname: str):
        raise NotImplementedError('Have patience...')

    def normalize(self, method: str='scalar', degree: int=None, **kwargs) -> None:
        """See documentation for spectrum_overload.Spectrum.normalize"""
        self.spectrum = self.spectrum.normalize(method, degree, **kwargs)
        self.flux = self.spectrum.flux
    
    def to_wavelength(self, new_wavelength: data_type) -> None:
        self.spectrum.interpolate1d_to(new_wavelength)

    def noise_level(self, interval: interval_type) -> number_type:
        if interval[0] >= interval[1]:
            raise ValueError('Lower limit should be given first')
        idx = (interval[0] <= self.wavelength) & (self.wavelength <= interval[1])
        return np.std(self.flux[idx])

    def noise_levels(self, intervals: intervals_type) -> np.ndarray:
        noise_levels = np.zeros(len(intervals))
        for i, interval in enumerate(intervals):
            noise_levels[i] = self.noise_level(interval)
        return noise_levels

    def mean_noise_level(self, intervals: intervals_type) -> number_type:
        noise_levels = self.noise_levels(intervals)
        return sum(noise_levels)/len(noise_levels)

    def plot(self, **kwargs) -> None:
        self.spectrum.plot(**kwargs)

    def __sub__(self, other):
        if not isinstance(other, SpectrumAbstraction):
            raise ValueError('Can not subtract type {} from Spectrum class'.format(type(other)))
        return SpectrumAbstraction.from_spectrum(self.spectrum - other.spectrum)


if __name__ == "__main__":
    from astropy.io import fits
    from glob import glob
    import numpy as np
    from .utils import get_wavelength
    file = glob('*.fits')[0]

    d = fits.open(file)[0]
    flux = d.data
    wavelength = get_wavelength(d.header)

    spec = spc(xaxis=wavelength, flux=flux)
    spec.wav_select(wavelength[0]*1.1, wavelength[-1])
    spec = spec.normalize('quadratic')

    s1 = SpectrumAbstraction(spec.xaxis, spec.flux)
    sigma = spec.flux / 42
    noise = np.random.normal(0, sigma)
    s2 = SpectrumAbstraction(spec.xaxis, spec.flux + noise)
    s3 = s1-s2
