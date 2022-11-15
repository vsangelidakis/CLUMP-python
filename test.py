import clumpGenerator_Ferellec_McDowell
import numpy as np

CLUMP = clumpGenerator_Ferellec_McDowell.clumpGenerator_Ferellec_McDowell(
    stl_dir="Rock.stl",
    rstep=0.001,
    rmin=0.01,
    dmin=0.01,
    pmax=1.0,
    isShuffled=True)

np.savetxt("clump_positions", CLUMP.clump.positions, delimiter=';')
np.savetxt("clump_radii", CLUMP.clump.radii, delimiter=';')

