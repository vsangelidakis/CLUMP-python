import clumpGenerator_Ferellec_McDowell
import numpy as np
from time import perf_counter

t1 = perf_counter()

CLUMP = clumpGenerator_Ferellec_McDowell.clumpGenerator_Ferellec_McDowell(
    stl_dir="Rock.stl",
    rstep=0.001,
    rmin=0.01,
    dmin=0.01,
    pmax=1.0,
    output_dir='my_clump',
    isShuffled=True)

print(perf_counter()-t1)

