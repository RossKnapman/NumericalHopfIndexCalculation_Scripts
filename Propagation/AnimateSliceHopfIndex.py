import numpy as np
from discretisedfield import Field
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
import os
import sys

sys.path.append('..')
from Utils import PlottingFunctions, QuantityReader


plt.style.use('seaborn-v0_8-colorblind')


#########################
# Constant Declarations #
#########################

SIMULATION_DIRECTORY = 'Data'
PARAMS_FILE          = 'params'


################################
# Collect Files to Be Rendered #
################################

simulation_files = []

for simulation_file in os.listdir(SIMULATION_DIRECTORY):
    if simulation_file.startswith('m') and simulation_file.endswith('.ovf'):
        simulation_files.append(simulation_file)

simulation_files = sorted(simulation_files)
assert len(simulation_files) > 0


###############
# Set Up Plot #
###############

fig, ax = plt.subplots()

initial_magnetisation = Field.from_file(SIMULATION_DIRECTORY + '/m000000.ovf').array

Nx, Ny, Nz,  _ = initial_magnetisation.shape
Lx             = Ly = QuantityReader.read_quantity('Lxy', PARAMS_FILE)
Lz             = Lx * Nz / Ny
discretisation = Lx / Nx

colour_plot = ax.imshow(
    initial_magnetisation[:, Ny//2, :, 2].T,  # Imshow treats first dimension as y-axis so transpose
    animated = True,
    origin   = 'lower',
    extent   = [0, Lx, 0, Lz],
    cmap     = 'RdBu_r'
)

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$z$')
ax.set_aspect('equal')


####################
# Create Rectangle #
####################

hopf_index_region_length   = QuantityReader.read_quantity('HopfIndexRegionLength', PARAMS_FILE)
hopf_index_region_position = QuantityReader.read_quantity('HopfIndexRegionPosition', PARAMS_FILE)
rectangle_y                = (hopf_index_region_position - hopf_index_region_length/2) * Lz
rectangle_height           = hopf_index_region_length * Lz

rect = Rectangle(
    (0, rectangle_y),
    Lx, rectangle_height,
    linewidth=2, edgecolor='limegreen', facecolor='none'
)

ax.add_patch(rect)


###########################
# Magnetisation Animation #
###########################

def update_anim(frame):
    print(frame / len(simulation_files), end='\r')
    colour_plot.set_array(
        Field.from_file(SIMULATION_DIRECTORY + f'/m{frame:06d}.ovf').array[:, Ny//2, :, 2].T,
    )

anim = animation.FuncAnimation(
    fig,
    update_anim,
    iter(range(len(simulation_files))),
    save_count=len(simulation_files)
)

anim.save('Slice.mp4', fps=25, writer='ffmpeg')


##########################
# Hopf Index Calculation #
##########################

hopf_index_start_z = int(np.round(Nz * (hopf_index_region_position - hopf_index_region_length/2)))
hopf_index_end_z = int(np.round(Nz * (hopf_index_region_position + hopf_index_region_length/2)))

hopf_indices_solidangle  = np.zeros(len(simulation_files), dtype=float)
hopf_indices_twopointfd  = np.zeros(len(simulation_files), dtype=float)
hopf_indices_fivepointfd = np.zeros(len(simulation_files), dtype=float)
hopf_indices_fourier     = np.zeros(len(simulation_files), dtype=float)

for file_index in range(len(simulation_files)):

    #########################
    # Real Space Hopf Index #
    #########################

    hopf_index_density_solidangle = Field.from_file(SIMULATION_DIRECTORY + f'/ext_hopfindexdensity_solidangle{file_index:06d}.ovf').array
    hopf_indices_solidangle[file_index] = -discretisation**3 * np.sum(hopf_index_density_solidangle[:, :, hopf_index_start_z:hopf_index_end_z])

    hopf_index_density_twopointfd = Field.from_file(SIMULATION_DIRECTORY + f'/ext_hopfindexdensity_twopointstencil{file_index:06d}.ovf').array
    hopf_indices_twopointfd[file_index] = -discretisation**3 * np.sum(hopf_index_density_twopointfd[:, :, hopf_index_start_z:hopf_index_end_z])

    hopf_index_density_fivepointfd = Field.from_file(SIMULATION_DIRECTORY + f'/ext_hopfindexdensity_fivepointstencil{file_index:06d}.ovf').array
    hopf_indices_fivepointfd[file_index] = -discretisation**3 * np.sum(hopf_index_density_fivepointfd[:, :, hopf_index_start_z:hopf_index_end_z])

    ############################
    # Fourier Space Hopf Index #
    ############################

    F_array = Field.from_file(SIMULATION_DIRECTORY + f'/ext_emergentmagneticfield_solidangle{file_index:06d}.ovf').array[:, :, hopf_index_start_z:hopf_index_end_z]
    F_array[:, :, :, 0] *= discretisation**2
    F_array[:, :, :, 1] *= discretisation**2
    F_array[:, :, :, 2] *= discretisation**2

    Nx, Ny, Nz, _ = F_array.shape

    kx = np.fft.fftfreq(Nx)
    ky = np.fft.fftfreq(Ny)
    kz = np.fft.fftfreq(Nz)

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K = np.zeros((Nx, Ny, Nz, 3))
    K[:, :, :, 0] = KX
    K[:, :, :, 1] = KY
    K[:, :, :, 2] = KZ

    Fx_k = np.fft.fftn(F_array[:, :, :, 0])
    Fy_k = np.fft.fftn(F_array[:, :, :, 1])
    Fz_k = np.fft.fftn(F_array[:, :, :, 2])

    F_k = np.zeros((Nx, Ny, Nz, 3), dtype=np.complex_)
    F_k[:, :, :, 0] = Fx_k
    F_k[:, :, :, 1] = Fy_k
    F_k[:, :, :, 2] = Fz_k

    F_mk = np.conj(F_k)

    # Elementwise dot product
    k2 = np.einsum('ijkl,ijkl->ijk', K, K)

    summand = np.einsum('ijkl,ijkl->ijk', F_mk, np.cross(K, F_k, axis=-1)) / k2
    summand[np.where(np.isclose(k2, 0.0))] = 0

    hopf_indices_fourier[file_index] = np.imag(np.sum(summand)) / (2*np.pi * Nx*Ny*Nz)



###################
# Hopf Index Plot #
###################

fig, ax = plt.subplots()

simulation_time = QuantityReader.read_quantity('SimulationTime', PARAMS_FILE)
times = np.linspace(0, simulation_time, len(hopf_indices_solidangle))


fig, ax = plt.subplots()

fivepointfd_plot, = ax.plot(times[0], hopf_indices_fivepointfd[0], animated=True, label='ext_hopfindex_fivepointstencil')
twopointfd_plot, = ax.plot(times[0], hopf_indices_twopointfd[0], animated=True, label='ext_hopfindex_twopointstencil')
solidangle_plot, = ax.plot(times[0], hopf_indices_solidangle[0], animated=True, label='ext_hopfindex_solidangle')
fourier_plot, = ax.plot(times[0], hopf_indices_fourier[0], animated=True, label='ext_hopfindex_solidanglefourier')

ax.set_ylim(0.9*np.min(hopf_indices_solidangle), 1.1*np.max(hopf_indices_solidangle))
ax.set_xlim(0, np.max(times))
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$H$')
ax.legend(loc='upper left')

def update_anim(frame):
    fivepointfd_plot.set_data(times[:frame+1], hopf_indices_fivepointfd[:frame+1]),
    twopointfd_plot.set_data(times[:frame+1], hopf_indices_twopointfd[:frame+1]),
    solidangle_plot.set_data(times[:frame+1], hopf_indices_solidangle[:frame+1]),
    fourier_plot.set_data(times[:frame+1], hopf_indices_fourier[:frame+1]),


anim = animation.FuncAnimation(
    fig,
    update_anim,
    iter(range(len(times))),
    save_count=len(times)
)

anim.save('HopfIdx.mp4', fps=25, writer='ffmpeg')
