# -*- coding: future_fstrings -*-

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib


_COLS = ('wavelength', 'nlines', 'depth', 'fwhm',
        'EW', 'EWerr', 'amplitude', 'sigma', 'mean')


class ARES:
    def __init__(self, *arg, **kwargs):
        self._config_created = False
        self.kwargs = kwargs
        self._create_config()
        self.outputfile = self.kwargs.get('fileout', 'test.ares')

    @classmethod
    def from_config(cls, fname):
        with open(fname) as lines:
            kwargs = dict()
            for line in lines:
                line = line.split('=')
                line = list(map(lambda s: s.replace(' ', ''), line))
                kwargs[line[0]] = kwargs[1]
        return cls(**kwargs)

    def _create_config(self):
        fout =  f"specfits='{self.kwargs.get('specfits')}'\n"
        fout += f"readlinedat='{self.kwargs.get('readlinedat', 'cdo.dat')}'\n"
        fout += f"fileout='{self.kwargs.get('fileout', 'aresout.dat')}'\n"
        fout += f"lambdai={self.kwargs.get('lambdai', 3000)}\n"
        fout += f"lambdaf={self.kwargs.get('lambdaf', 8000)}\n"
        fout += f"smoothder={self.kwargs.get('smoothder', 6)}\n"
        fout += f"space={self.kwargs.get('space', 3.0)}\n"
        fout += f"rejt={self.kwargs.get('rejt', '3;5764,5766,6047,6052,6068,6076')}\n"
        fout += f"lineresol={self.kwargs.get('lineresol', 0.05)}\n"
        fout += f"miniline={self.kwargs.get('miniline', 2)}\n"
        fout += f"plots_flag={self.kwargs.get('plots_flag', 0)}\n"
        fout += f"rvmask='{self.kwargs.get('rvmask', '3,6021.8,6024.06,6027.06,6024.06,20')}'\n"
        with open('mine.opt', 'w') as f:
            f.writelines(fout)
        self._config_created = True

    def run(self, verbose=False):
        if not self._config_created:
            self._create_config()
        if verbose:
            print('Running ARES...')
        os.system('ARES > /dev/null')
        if verbose:
            print(f'Done! Result saved in {self.kwargs.get("fileout", "aresout.dat")}')

    @staticmethod
    def read_output(fname: str):
        return ARESOutput(fname)

    @staticmethod
    def get_rv(fname: str='logARES.txt') -> float:
        with open(fname, 'r') as lines:
            for line in lines:
                if line.startswith('Velocidade radial'):
                    break
        rv = line.rpartition(':')[-1]
        return float(rv)


class ARESOutput:
    def __init__(self, fname, *args, **kwargs):
        self.fname = fname
        self.df = pd.read_csv(self.fname, sep=r'\s+', header=None)
        self.df.columns = _COLS

    def percent_diff(self, other, col):
        """Find the percent difference between two ARES output for a given column.
        Is given by:
            (self.df.col - other.df.col) / self.df.col * 100

        Input
        -----
        other : ARESOutput object
            The result from another spectrum
        col : str
            The col for which to calculate the difference
        """
        if not isinstance(other, ARESOutput):
            raise TypeError(f'other is of type {type(other)} which is not compatible for this method')
        if not col in _COLS:
            raise ValueError(f'The following columns are allowed: {_COLS}')
        p = (self.df[col] - other.df[col]) / self.df[col] * 100
        return p.values

    def plot(self, col1, col2=None, *args, **kwargs):
        if col2 is None:
            col2 = col1
            col1 = 'wavelength'
        if not (col1 in _COLS) or not (col2 in _COLS):
            raise ValueError(f'The following columns are allowed: {_COLS}')

        plt.plot(self.df[col1], self.df[col2], 'o', *args, **kwargs)
        plt.xlabel(col1)
        plt.ylabel(col2)

    def mse(self, other, col):
        if not isinstance(other, ARESOutput):
            raise TypeError(f'other is of type {type(other)} which is not compatible for this method')
        if not col in _COLS:
            raise ValueError(f'The following columns are allowed: {_COLS}')
        N = max((len(self.df), len(other.df)))
        df = self.df.join(other.df, how='outer', lsuffix='_l', rsuffix='_r')
        return 1/N * np.sqrt((np.sum((df[col+'_l'] - df[col+'_r'])**2)))


def get_result(*args, **kwargs):
    if not os.path.exists(kwargs.get('fileout')):
        a = ARES(*args, **kwargs)
        a.run(verbose=True)
        return ARES.read_output(kwargs.get('fileout'))
    return ARES.read_output(kwargs.get('fileout'))


def create_fname(spectrum, smoothder, space, star='sun'):
    if 'espresso' in spectrum.lower():
        folder = 'ESPRESSO'
    elif 'pepsi' in spectrum.lower():
        folder = 'PEPSE'
    elif 'harps' in spectrum.lower():
        folder = 'HARPS'
    return f'../data/ARES/{folder}/{star}/smooth{smoothder}_space{space}.ares'


def setup_dirs():
    pathlib.Path("../data/ARES/ESPRESSO/sun").mkdir(parents=True, exist_ok=True)
    pathlib.Path("../data/ARES/PEPSI/sun").mkdir(parents=True, exist_ok=True)
    pathlib.Path("../data/ARES/HARPS/sun").mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    setup_dirs()
    spectra = ('../daniel/sun_espresso_s1d_rv.fits', '../daniel/sun_pepsi_1d_rv.fits')
    smoothders = range(1, 10)
    spaces = np.arange(1, 8, 0.1)
    for spectrum in spectra:
       for smoothder in smoothders:
           for space in spaces:
               config = {'specfits': spectrum,
                         'readlinedat': '../daniel/linelist_damp.rdb',
                         'fileout': create_fname(spectrum, smoothder, space),
                         'smoothder': smoothder,
                         'space': space}

    output_espresso = get_result(**{'fileout': '../data/ARES/ESPRESSO/sun/smooth5_space3.2.ares'})
    output_pepsi = get_result(**{'fileout': '../data/ARES/PEPSI/sun/smooth5_space3.2.ares'})

    mse_EW = output_espresso.mse(output_pepsi, 'EW')
