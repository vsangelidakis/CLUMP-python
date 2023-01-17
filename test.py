import clumpGenerator_Ferellec_McDowell, clumpGenerator_Euclidean_Distance_Transform
from time import perf_counter

t1 = perf_counter()
clumpGenerator_Ferellec_McDowell.clumpGenerator_Ferellec_McDowell(
    stl_dir="Rock.stl",
    rstep=0.001,
    rmin=0.01,
    dmin=0.01,
    pmax=1.0,
    output_dir='my_clump_fd',
    isShuffled=True)

print(f"Ferellec McDowell:\t{perf_counter() - t1}")


t2 = perf_counter()
clumpGenerator_Euclidean_Distance_Transform.clumpGenerator_Euclidean_Distance_Transform(
    stl_dir="Rock.stl",
    N=24,
    rMin=0.0,
    div=102,
    overlap=0.6,
    output_dir='my_clump_edt')

print(f"Euclidean Distance Transform:\t{perf_counter() - t2}")
