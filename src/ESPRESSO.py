from .spectrum import SpectrumAbstraction


class ESPRESSO(SpectrumAbstraction):
    def __init__(self, wavelength, flux, *args, **kwargs):
        super().__init__(wavelength, flux, *args, **kwargs)

    