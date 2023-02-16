import numpy as np
import trimesh
import utils.PatchNormals as PatchNormals
import utils.RigidBodyParameters as RigidBodyParameters
import utils.STLReader as STLReader
from scipy.ndimage import distance_transform_edt


class Clump:
    def __init__(self):
        self.positions = np.array([]).reshape((0, 3))
        self.radii = np.array([]).reshape((0, 1))


class Ferellec_McDowell(Clump):
    def __init__(self, stl_dir, rstep, rmin, dmin, pmax, output_dir=None, isShuffled=True):
        """
        2020 © V. Angelidakis, S. Nadimi, M. Otsubo, S. Utili.
        [1] Ferellec, J.F. and McDowell, G.R., 2010. Granular Matter, 12(5), pp.459-467. DOI 10.1007/s10035-010-0205-8

        The main concept of this methodology:
        1. We import the surface mesh of a particle.
        2. We calculate the normal of each vertex pointing inwards.
        3. For a random vertex on the particle surface, we start creating
             tangent spheres with incremental radii along the vertex normal,
             starting from 'rmin', with a step of 'rstep', until they meet the
             surface of the particle.
        4. We select a new vertex randomly, which has a distance larger
           than 'dmin' from the existing spheres and do the same.
        5. When a percentage 'pmax' of all the vertices is used to generate
           spheres, the generation procedure stops.
        -  An optional 'seed' parameter is introduced, to generate reproducible
           clumps.

        :param stl_dir:
            Directory of stl file, used to generate spheres
        :param rstep:
            Step used to increase the radius in each iteration, until the generated sphere meets another point
            of the particle.
        :param rmin:
            Minimum radius of sphere to be generated. For coarse meshes, the actual minimum radius might be >rmin.
        :param dmin:
            Minimum allowed distance between new vertex of the surface mesh and existing spheres. If left zero, this
            distance is not cheched.
        :param pmax:
            Percentage of vertices which will be used to generate spheres. The selection of vertices is random.
        :param output_dir:
            File name for output of the clump in .txt form	(optional)*. If output_dir = None assigned, a .txt output
            file is not created.
        :param isShuffled:
        """

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

        if output_dir is not None:
            np.savetxt(output_dir, np.asarray(np.hstack((self.clump.positions, self.clump.radii))),
                       delimiter=",")  # In PyCharm this line seems to have an error but it does not.

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
                dcur = np.min(
                    np.sqrt(np.square(x - self.clump.positions[:, 0].reshape(self.clump.positions.shape[0], 1))
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


class Euclidean_Distance_Transform(Clump):
    def __init__(self, stl_dir, N, rMin, div, overlap, output_dir=None):
        """
        Clump generator using the Euclidean map for voxelated, 3D particles
        2020 © V. Angelidakis, S. Nadimi, M. Otsubo, S. Utili.

        The main concept of this methodology:
        1. We import the surface mesh of a particle.
        2. We transform the mesh into a voxelated representation, i.e. a binary
           3D image, where each voxel belonging to the particle is equal to zero.
        3. The Euclidean distance transform of the 3D image is computed and
           the radius of the largest inscribed sphere is found as the maximum
           value of the Euclidean transform of the voxelated image.
        4. The voxels corresponding to the inscribed sphere are then set equal to
           one. This methodology can also generate overlapping spheres, if only a
           percentage of the voxels of each new sphere are set equal to one,
           instead of all of them.
        5. This process is repeated until a user-defined number of spheres 'N' is
           found or until the user-defined minimum radius criterion has been met,
           as the spheres are generated in decreasing sizes.

        :param stl_dir:
            File name of the STL file used to generate clumps
        :param N:
            [1,inf)  Larger N will lead to a larger number of spheres
        :param rMin:
            (0,inf)  Larger rMin will lead to a smaller number of spheres
        :param div:
            (5,inf]  Larger div will lead to better shape resolution in voxel space
        :param overlap:
            [0,1)	 Larger overlap will lead to larger spheres overall
        :param output_dir:
            File name of the for output of the clump in .txt form (optional)
        """

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


class Favier:
    def __init__(self, stl_dir, N, chooseDistance, output_dir=None):
        """
        2021 © V. Angelidakis, S. Nadimi, M. Otsubo, S. Utili.
        [1] Favier, J.F., Fard, M.H., Kremmer, M. and Raji, A.O., 1999. Engineering Computations: Int J for Computer-Aided Engineering, 16(4), pp.467-480.

        :param stl_dir:
            Input geometry, given in one of the formats below:
                        1. Directory of .stl file (for surface meshes)
                        2. Directory of .mat file (for binary voxelated images)
                        3. Struct with fields {vertices,faces} (for surface meshes)
                        4. Struct with fields {img,voxel_size} (for binary voxelated images)
                           where
                            - img:			[Nx x Ny x Nz] voxelated image
                            - voxel_size:	[1x3] voxel size in Cartesian space
        :param N:
            Number of spheres to be generated.

        :param chooseDistance:
            Preferred method to specify the radii of the spheres, which can be either the minimum ('min'),
            the average ('avg') or the maximum ('max') distance of the vertices within the span of interest.

        :param output_dir:
            File name of the for output of the clump in .txt form (optional)
        """

        super().__init__()

        self.clump = Clump()

        self.stl_dir = stl_dir
        self.N = N
        self.chooseDistance = chooseDistance

        self.F, self.P = STLReader.read_stl(self.stl_dir)
        self.RBP = RigidBodyParameters.RBP(self.F, self.P)

        self.mesh = trimesh.load_mesh(self.stl_dir)

        self.generate_clump()

        if output_dir is not None:
            np.savetxt(output_dir, np.asarray(np.hstack((self.clump.positions, self.clump.radii))),
                       delimiter=",")

    def generate_clump(self):
        rot = self.RBP.PAI
        # Singular value decomposition implementations in numpy and MATLAB give different results due to the
        # freedom of choosing the basis vectors. To make them same I added two lines below. They can be removed
        # without loss of generality.
        rot *= -1
        rot[:, 1] *= -1

        # Transform rotation matrix to align longest axis along X direction
        rot[:, [2, 0]] = rot[:, [0, 2]]

        self.P = self.P - self.RBP.centroid
        self.P = self.P @ rot

        X_extremas = np.array([np.min(self.P[:, 0]), np.max(self.P[:, 0])])
        a, b = X_extremas[0], X_extremas[1]

        nSegments = self.N

        endPoints = np.linspace(a, b, nSegments + 1)
        start = endPoints[0:-1]
        stop = endPoints[1::]
        midPoints = stop - (stop[0] - start[0]) / 2

        minDistance = np.zeros(midPoints.size)
        minDx = np.zeros(midPoints.size)

        p_struct = {}  # MATLAB's struct is equivalent to Python's dictionary
        for i in range(midPoints.size):
            temp_arr = np.array([])  # I initialize an emtpy array to stack the values required below

            # Find vertices within each sector
            for j in range(self.P.shape[0]):  # for each point of the particle surface (in principal axes)
                if endPoints[i] <= self.P[j, 0] <= endPoints[i + 1]:
                    temp_arr = np.append(temp_arr, self.P[j, 0:3])  # I concatante the (3, 1) arrays
            p_struct[i] = temp_arr.reshape(temp_arr.size // 3, 3)  # then I reshape the temp_arr to add it to p_struct

            # Find closest distance of midpoint to any surface vertex
            xM, yM, zM = midPoints[i], 0, 0
            x1 = endPoints[0]
            x2 = endPoints[-1]

            # Closest distance between midpoint and particle surface
            minDistance[i] = np.min(
                np.sqrt(
                    np.square(self.P[:, 0] - xM) + np.square(self.P[:, 1] - yM) + np.square(self.P[:, 2] - zM)))

            # Closest distane between midpoint and particle X limits
            minDx[i] = np.min((np.abs(x1 - xM), np.abs(x2 - xM)))

        radius = np.zeros(midPoints.size)
        for i in range(midPoints.size):  # I skipped the isempty query.
            distance = np.sqrt(np.square(p_struct[i][:, 0] - midPoints[i]) + np.square(p_struct[i][:, 1]) + np.square(
                p_struct[i][:, 2]))

            # I wrote the switch-case expression in old-school fashion for backward compability
            if self.chooseDistance == "min":
                radius[i] = np.min(distance)
            elif self.chooseDistance == "avg":
                radius[i] = np.mean(distance)
            elif self.chooseDistance == "max":
                radius[i] = np.max(distance)

            radius[i] = np.min((radius[i], minDx[i]))

            self.clump.positions = np.vstack((self.clump.positions, np.array([midPoints[i], 0, 0])))
            self.clump.radii = np.vstack((self.clump.radii, radius[i]))

        # Transform the mesh and the clump coordinates back to the initial (non-principal) system
        self.P = self.P @ np.transpose(rot)
        self.P += self.RBP.centroid

        self.clump.positions = self.clump.positions @ np.transpose(rot)
        self.clump.positions += self.RBP.centroid

