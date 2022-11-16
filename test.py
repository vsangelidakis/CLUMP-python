import clumpGenerator_Ferellec_McDowell
import numpy as np

CLUMP = clumpGenerator_Ferellec_McDowell.clumpGenerator_Ferellec_McDowell(
    stl_dir="Rock.stl",
    rstep=0.001,
    rmin=0.01,
    dmin=0.01,
    pmax=1.0,
    isShuffled=True)

np.savetxt('clump_info.txt', np.asarray(np.hstack((CLUMP.clump.positions, CLUMP.clump.radii))), delimiter=",")
