from qubic import QubicScene
from SynthBeam import myinstrument
import glob
import healpy as hp


scene = QubicScene(256)
inst = myinstrument.QubicInstrument(filter_nu=150e9)

