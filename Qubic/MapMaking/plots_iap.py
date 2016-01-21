
from qubic.io import read_map
mapsin = read_map('input_map_convolved_pointing_sigma_0.0_seed_22701.fits')
mapsout = read_map('reconstructed_map_pointing_sigma_0.0_seed_22701.fits')

def display(input, msg, iplot=1, reso=5, Trange=[100, 5, 5], xsize=800):
    out = []
    for i, (kind, lim) in enumerate(zip('IQU', Trange)):
        map = input[..., i]
        out += [hp.gnomview(map, rot=center, reso=reso, xsize=xsize, min=-lim,
                            max=lim, title=msg + ' ' + kind,
                            sub=(3, 3, iplot + i), return_projected_map=True)]
    return out

center = equ2gal(racenter, deccenter+10)

reso=8
xsize=400
mp.clf()
display(mapsin, 'Original map', iplot=1, reso=reso, xsize=xsize)
display(mapsout, 'Reconstructed map', iplot=4, reso=reso, xsize=xsize)
display(mapsin-mapsout, 'Residual map', iplot=7, reso=reso, xsize=xsize)

hp.mollview(mapsin[:,1]-mapsout[:,1],min=-5,max=5)