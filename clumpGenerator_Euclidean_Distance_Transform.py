import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt
from utils import RigidBodyParameters, STLReader


class Clump:
    def __init__(self):
        self.positions = np.array([]).reshape((0, 3))
        self.radii = np.array([]).reshape((0, 1))


class clumpGenerator_Euclidean_Distance_Transform(Clump):
    def __init__(self, stl_dir, N, rMin, div, overlap, output_dir=None):
        super().__init__()

        self.clump = Clump()

        self.stl_dir = stl_dir
        self.N = N
        self.rMin = rMin
        self.div = div
        self.overlap = overlap

        self.F, self.P = STLReader.read_stl(self.stl_dir)
        self.RBP = RigidBodyParameters.RBP(self.F, self.P)

        self.mesh = trimesh.load_mesh(self.stl_dir)

        self.generate_clump()

        if output_dir is not None:
            np.savetxt(output_dir, np.asarray(np.hstack((self.clump.positions, self.clump.radii))),
                       delimiter=",")

    def generate_clump(self):
        # %% Calculate extreme coordinates & centroid of the AABB of the particle
        minX, minY, minZ = np.min(self.P[:, 0]), np.min(self.P[:, 1]), np.min(self.P[:, 2])
        maxX, maxY, maxZ = np.max(self.P[:, 0]), np.max(self.P[:, 1]), np.max(self.P[:, 2])
        aveX, aveY, aveZ = np.mean((minX, maxX)), np.mean((minY, maxY)), np.mean((minZ, maxZ))

        # %% Center the particle to the centroid of its AABB
        self.P[:, 0] -= aveX
        self.P[:, 1] -= aveY
        self.P[:, 2] -= aveZ

        # div:		Division number along the shortest edge of the AABB during
        # %				voxelisation (resolution). If not given, div=50 (default
        # %				value in iso2mesh).
        min_AABB = np.min((np.abs(maxX - minX), np.abs(maxY - minY), np.abs(maxZ - minZ)))
        voxel_size = min_AABB / self.div

        def matrix_zero_padding(matrix, pn=2):
            # matrix is 3 dimensional
            # pn: padding number indicating that how many zeros will be added on one edge
            temp_mat = np.zeros(matrix.shape + np.array([1, 1, 1]) * 2 * pn)
            temp_mat[pn:-pn, pn:-pn, pn:-pn] = matrix
            return temp_mat

        img_temp = self.mesh.voxelized(pitch=voxel_size, method="subdivide").fill()
        intersection = matrix_zero_padding(np.array(
            img_temp.matrix))  # % Expand the image by 2 voxels in each direction, to ensure the boundary voxels are false (zeros).

        #  %% Dimensions of the new image
        halfSize = np.array(intersection.shape) / 2
        dx, dy, dz = np.meshgrid(np.arange(1, intersection.shape[1] + 1), np.arange(1, intersection.shape[0] + 1),
                                 np.arange(1, intersection.shape[2] + 1))  # be careful about the order

        #  %% Calculate centroid of the voxelated image
        centroid = self.RBP.centroid

        for _ in range(self.N):
            # edtImage = distance_transform_edt(np.logical_not(intersection), return_distances=True)
            edtImage = distance_transform_edt(intersection, return_distances=True)
            radius = np.max(edtImage)

            if radius < self.rMin:
                print("['The mimimum radius rMin=',num2str(rMin),' has been met using ', num2str(k-1),' spheres']")

            xyzCenter = np.argwhere(edtImage == radius)

            dists = np.sqrt(np.sum(np.power(centroid - xyzCenter, 2), axis=1))
            i = np.argmax(dists)

            sph = np.sqrt(
                (dx - xyzCenter[i, 1]) ** 2 + (dy - xyzCenter[i, 0]) ** 2 + (dz - xyzCenter[i, 2]) ** 2) > (
                              1 - self.overlap) * radius

            intersection = np.logical_and(intersection, sph)

            xyzC = xyzCenter[i] - halfSize + 1

            self.clump.positions = np.vstack((self.clump.positions, xyzC * voxel_size))
            self.clump.radii = np.vstack((self.clump.radii, radius * voxel_size))
