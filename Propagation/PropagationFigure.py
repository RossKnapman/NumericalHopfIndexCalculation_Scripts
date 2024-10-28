import pyvista as pv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from discretisedfield import Field
from PIL import Image
import os
import sys

sys.path.append('..')
from Utils import PlottingFunctions, QuantityReader


#################
# Set Up Figure #
#################

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['axes.labelsize'] = 18
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14
plt.style.use('seaborn-v0_8-colorblind')

fig = plt.figure(figsize=(7, 7))
gs  = fig.add_gridspec(2, 2, height_ratios=[1, 1])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])



#########################
# Constant Declarations #
#########################

SIMULATION_INDEX     = 140
WINDOW_SIZE          = [1000, 1000]
SIMULATION_DIRECTORY = 'Data'
PARAMS_FILE          = 'params'
CAMERA_POSITION      = (35, 35, 30)
CAMERA_FOCAL_POINT   = (10, 10, 13)
CAMERA_VIEW_UP       = (0, 0, 1)


################
# Set Up Scene #
################

plotter = pv.Plotter(window_size=WINDOW_SIZE, off_screen=True)
plotter.enable_anti_aliasing('ssaa')
data = pv.read(SIMULATION_DIRECTORY + f'/m{SIMULATION_INDEX:06}.vts')
Nx, Ny, Nz = data.dimensions

bounding_box = data.outline()
plotter.add_mesh(bounding_box, line_width=2, color='black')

plotter.camera_position = [
    CAMERA_POSITION,
    CAMERA_FOCAL_POINT,
    CAMERA_VIEW_UP
]


##################
# Plot Preimages #
##################

preimage_azimuthal_angles = np.linspace(0, 2*np.pi, 8)
for i, angle in enumerate(preimage_azimuthal_angles):

    PlottingFunctions.plot_preimage(
        data,
        plotter,
        (np.cos(angle), np.sin(angle), 0),
        preimage_closeness = 0.95
    )


#############################
# Plot Slice Through System #
#############################

data['mz'] = data['m'][:, 2]
single_slice = data.slice(normal=[1, 0, 0])
plotter.add_mesh(
    single_slice,
    scalars         = 'mz',
    clim            = [-1, 1],
    opacity         = 0.4,
    cmap            = 'RdBu_r',
    show_scalar_bar = False
)

plotter.screenshot('/tmp/Preimages.png')
preimages_image = np.asarray(Image.open('/tmp/Preimages.png'))
ax1.imshow(preimages_image, interpolation='none')
ax1.axis('off')


#################
# 2D Slice Plot #
#################

magnetisation_array = Field.from_file(SIMULATION_DIRECTORY + f'/m{SIMULATION_INDEX:06}.ovf').array

Nx, Ny, Nz,  _ = magnetisation_array.shape
Lx             = Ly = QuantityReader.read_quantity('Lxy', PARAMS_FILE)
Lz             = Lx * Nz / Ny
discretisation = Lx / Nx

colour_plot = ax2.imshow(
    magnetisation_array[:, Ny//2, :, 2].T,  # Imshow treats first dimension as y-axis so transpose
    animated = True,
    origin   = 'lower',
    extent   = [0, Lx, 0, Lz],
    cmap     = 'RdBu_r',
    vmin     = -1.,
    vmax     = 1.
)

ax2.set_xticks([0, 5, 10])
ax2.set_yticks([0, 5, 10, 15, 20])
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$z$')
ax2.set_aspect('equal')

ax_cbar = inset_axes(
    ax2,
    width='5%',
    height='100%',
    loc='lower left',
    bbox_to_anchor = (1.1, 0., 1, 1),
    bbox_transform = ax2.transAxes,
    borderpad = 0
)

cb = fig.colorbar(colour_plot, cax=ax_cbar)
cb.ax.set_ylabel(r'$m_z$', rotation=0, loc='top', labelpad=20)


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

ax2.add_patch(rect)


#######################################
# Arrow Showing Propagation Direction #
#######################################

arrow = FancyArrowPatch(
    (0.5, 0.25),
    (0.5, 0.75),
    mutation_scale = 30,
    transform = ax2.transAxes,
    color='white'
)

ax2.add_patch(arrow)


###################
# Hopf Index Plot #
###################

simulation_files = []

for simulation_file in os.listdir(SIMULATION_DIRECTORY):
    if simulation_file.startswith('m') and simulation_file.endswith('.ovf'):
        simulation_files.append(simulation_file)

simulation_files = sorted(simulation_files)
assert len(simulation_files) > 0

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


simulation_time = QuantityReader.read_quantity('SimulationTime', PARAMS_FILE)
times = np.linspace(0, simulation_time, len(hopf_indices_solidangle))

ax3.plot(times, hopf_indices_fivepointfd, label='ext_hopfindex_fivepointstencil')
ax3.plot(times, hopf_indices_twopointfd, label='ext_hopfindex_twopointstencil')
ax3.plot(times, hopf_indices_solidangle, label='ext_hopfindex_solidangle')
ax3.plot(times, hopf_indices_fourier, label='ext_hopfindex_solidanglefourier')

ax3.legend()

ax3.set_xlabel('Time')
ax3.set_ylabel(r'$H$')
ax3.axhline(1.0, color='black', linestyle='dashed', alpha=0.5)
ax3.set_xlim(4, 10)

plt.tight_layout()
plt.savefig('Propagation.pdf', format='pdf')
