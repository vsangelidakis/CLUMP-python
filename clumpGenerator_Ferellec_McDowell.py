import numpy as np
from utils import RigidBodyParameters, STLReader, PatchNormals


class Clump:
    def __init__(self):
        self.positions = np.array([]).reshape((0, 3))
        self.radii = np.array([]).reshape((0, 1))


class clumpGenerator_Ferellec_McDowell(Clump):
    def __init__(self, stl_dir, rstep, rmin, dmin, pmax, isShuffled=True):
        super().__init__()

        self.clump = Clump()

        self.stl_dir = stl_dir
        self.F, self.P = STLReader.read_stl(self.stl_dir)

        self.RBP = RigidBodyParameters.RBP(self.F, self.P)
        self.N = PatchNormals.patch_normals(self.F, self.P)

        self.rstep = rstep
        self.rmin = rmin
        self.dmin = dmin
        self.pmax = pmax

        self.isShuffled = isShuffled

        self.generate_clump()

    def generate_clump(self):
        for i in range(self.P.shape[0]):
            if np.dot((self.P[i, :]) - self.RBP.centroid, self.N[i, :]) > 0:
                self.N[i, :] = -self.N[i, :]

        Pmax = range(len(self.P))

        if self.isShuffled:
            Vertices = np.random.permutation(len(self.P))
        else:
            Vertices = np.arange(len(self.P))
        tol = self.rmin / 1000

        iCount = 0  # since I am stacking the arrays the counter param is not necessary
        for _ in Pmax:
            i = Vertices[iCount]
            r = self.rmin
            reachedMaxRadius = False

            x, y, z = self.P[i, 0:3]
            n = self.N[i, :]

            if iCount > 0 and self.dmin > 0:
                dcur = np.min(np.sqrt(np.square(x - self.clump.positions[:, 0].reshape(self.clump.positions.shape[0], 1))
                                      + np.square(y - self.clump.positions[:, 1].reshape(self.clump.positions.shape[0], 1))
                                      + np.square(z - self.clump.positions[:, 2].reshape(self.clump.positions.shape[0], 1)))
                              - self.clump.radii)

                if dcur < self.dmin:
                    iCount += 1
                    continue

            while not reachedMaxRadius:
                sphMin = 1e15

                while sphMin > -tol:
                    xC = x + r * n[0]
                    yC = y + r * n[1]
                    zC = z + r * n[2]

                    distance = np.sqrt(np.square(self.P[:, 0] - xC)
                                       + np.square(self.P[:, 1] - yC)
                                       + np.square(self.P[:, 2] - zC))
                    sph = np.square(distance / r) - 1.0
                    sphMin = np.min(sph)

                    r += self.rstep

                reachedMaxRadius = True
                indMin = np.argmin(sph)  # index of the minimum

                pointInside = self.P[indMin, :]

                vAB = np.array([pointInside[0] - x, pointInside[1] - y, pointInside[2] - z])
                vAD = np.dot(vAB, n) / np.linalg.norm(n)

                AB = np.linalg.norm(vAB)
                AD = np.linalg.norm(vAD)

                radius = AB ** 2 / AD / 2

                xC = x + radius * n[0]
                yC = y + radius * n[1]
                zC = z + radius * n[2]

                self.clump.positions = np.vstack((self.clump.positions, np.array([xC, yC, zC]).reshape((1, 3))))
                self.clump.radii = np.vstack((self.clump.radii, radius))

            pcur = self.clump.radii.shape[0] / self.P.shape[0]
            if pcur < self.pmax:
                iCount += 1
            else:
                break
