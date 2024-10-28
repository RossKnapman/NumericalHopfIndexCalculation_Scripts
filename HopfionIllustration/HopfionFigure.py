import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image
from colorsys import hls_to_rgb
import re
import sys

sys.path.append('..')
from Utils import PlottingFunctions

plt.style.use('Style.mplstyle')
fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(14, 7.5))


#################
# Set Variables #
#################

SPHERE_POSITION              = (1.1, 1., 0)
SPHERE_RADIUS                = 0.1
CONE_LENGTH                  = 0.2
CONE_RADIUS                  = 0.05
AXIS_CYLINDER_LENGTH         = 0.2
AXIS_CYLINDER_RADIUS         = 0.01
AXIS_CONE_LENGTH             = 0.05
AXIS_CONE_RADIUS             = 0.025
FILENAME_MAGNETISATION       = 'Data/Hopfion.vts'
FILENAME_EMERGENT_FIELD      = 'Data/EmergentField.vts'
CAMERA_POSITION              = (2.08614, -0.808406, 2.22323)
CAMERA_FOCAL_POINT           = (0.581878, 0.502237, 0.52739)
CAMERA_VIEW_UP               = (-0.25, 0.2, 0.95)
WINDOW_SIZE                  = [1000, 800]
MAGNETISATION_STRIDE         = 7
EMERGENT_FIELD_STRIDE        = 10
EMERGENT_FIELD_STRIDE_QUIVER = 7
SLICE_OUTLINE_WIDTH          = 10
XY_PLANE_OUTLINE_COLOUR      = 'blue'
YZ_PLANE_OUTLINE_COLOUR      = 'red'
SPINES_LINEWIDTH             = 2
COMPACT_SUPPORT_COLOUR       = 'grey'
COMPACT_SUPPORT_STYLE        = 'dashed'


# Make the colour map's zero value actually white and not very faint blue
blues = mpl.colormaps['Blues'].resampled(256)
blues_zero_white = blues(np.linspace(0., 1., 256))
blues_zero_white[0] = [1., 1., 1., 1.]
colour_map = ListedColormap(blues_zero_white)


def quiver_plot(ax, field_slice, horizontal_axis, vertical_axis, QUIVER_STRIDE):

    x = np.arange(Nx)
    y = np.arange(Ny)
    X, Y = np.meshgrid(x, y)

    magnitudes = np.sqrt(np.einsum('ijk,ijk->ij', field_slice, field_slice))
    zero_mask = np.invert(np.isclose(magnitudes, 0.))
    in_plane_magnitudes = np.sqrt(field_slice[:, :, horizontal_axis]**2 \
        + field_slice[:, :, vertical_axis]**2)

    field_slice_normalised_horizontal = field_slice[:, :, horizontal_axis] / in_plane_magnitudes
    field_slice_normalised_vertical   = field_slice[:, :, vertical_axis] / in_plane_magnitudes

    # Get rid of any large numbers caused by dividing by zero
    field_slice_normalised_horizontal *= zero_mask
    field_slice_normalised_vertical *= zero_mask

    colour_array = np.zeros((*in_plane_magnitudes.shape, 4))
    colour_array[:, :, 3] = in_plane_magnitudes / np.max(in_plane_magnitudes)
    colour_array = colour_array[::QUIVER_STRIDE, ::QUIVER_STRIDE, :]
    colour_array = colour_array.reshape((colour_array.shape[0] * colour_array.shape[1], 4))

    ax.quiver(
        X[::QUIVER_STRIDE, ::QUIVER_STRIDE],
        Y[::QUIVER_STRIDE, ::QUIVER_STRIDE],
        field_slice_normalised_horizontal[::QUIVER_STRIDE, ::QUIVER_STRIDE],
        field_slice_normalised_vertical[::QUIVER_STRIDE, ::QUIVER_STRIDE],
        pivot='mid', headlength=20, headwidth=10, headaxislength=20, color=colour_array
    )

    ticks = list(range(0, 150, 50))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(axis='both', which='major')
    ax.set_aspect('equal')


######################################
# Set Up 3D Magnetisation Plot Scene #
######################################

plotter = pv.Plotter(window_size=WINDOW_SIZE, off_screen=True)
plotter.enable_anti_aliasing('ssaa')
data = pv.read(FILENAME_MAGNETISATION)
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

    PlottingFunctions.add_sphere_preimage(
        plotter,
        (np.cos(angle), np.sin(angle), 0),
        SPHERE_POSITION,
        SPHERE_RADIUS,
        CONE_LENGTH,
        CONE_RADIUS
    )

axs[0, 0].text(890, 750, r'$m_x$')
axs[0, 0].text(960, 570, r'$m_y$')
axs[0, 0].text(770, 530, r'$m_z$')

axs[0, 0].text(-0.15, 1, r'a)', transform=axs[0, 0].transAxes)
axs[0, 1].text(-0.15, 1, r'b)', transform=axs[0, 1].transAxes)
axs[0, 2].text(-0.15, 1, r'c)', transform=axs[0, 2].transAxes)
axs[1, 0].text(-0.15, 1, r'd)', transform=axs[1, 0].transAxes)
axs[1, 1].text(-0.15, 1, r'e)', transform=axs[1, 1].transAxes)
axs[1, 2].text(-0.15, 1, r'f)', transform=axs[1, 2].transAxes)


##############################
# Plot Slices Through System #
##############################

def single_slice(normal, outline_colour):

    single_slice = data.slice(normal=normal)

    mx = single_slice['m'][:, 0]
    my = single_slice['m'][:, 1]
    mz = single_slice['m'][:, 2]

    hls_to_rgb_vectorised = np.vectorize(hls_to_rgb)
    l = 0.5 * mz + 0.5
    h = np.arctan2(my, mx) / (2*np.pi)
    r, g, b, = hls_to_rgb_vectorised(h, l, 1.0)

    colours = np.zeros((len(r), 3))
    colours[:, 0] = r
    colours[:, 1] = g
    colours[:, 2] = b

    single_slice['colours'] = colours
    plotter.add_mesh(
        single_slice,
        scalars         = colours,
        rgb             = True,
        show_scalar_bar = False,
        opacity         = 0.5
    )

    # Add outline for slice
    outline = single_slice.outline()
    plotter.add_mesh(outline, line_width=SLICE_OUTLINE_WIDTH, color=outline_colour)


single_slice([0, 0, 1], XY_PLANE_OUTLINE_COLOUR)
single_slice([1, 0, 0], YZ_PLANE_OUTLINE_COLOUR)


#################
# Legend Sphere #
#################

sphere = pv.Sphere(
    center           = (SPHERE_POSITION[0], SPHERE_POSITION[1], SPHERE_POSITION[2]),
    radius           = SPHERE_RADIUS,
    theta_resolution = 50,
    phi_resolution   = 50
)

x, y, z = sphere.points[:, 0], sphere.points[:, 1], sphere.points[:, 2]

hls_to_rgb_vectorised = np.vectorize(hls_to_rgb)
l = 0.5 * (z - SPHERE_POSITION[2]) / SPHERE_RADIUS + 0.5
h = np.arctan2(y - SPHERE_POSITION[1], x - SPHERE_POSITION[0]) / (2*np.pi)
r, g, b, = hls_to_rgb_vectorised(h, l, 1.0)

sphere['r'] = r
sphere['g'] = g
sphere['b'] = b

colours = np.zeros((len(r), 3))
colours[:, 0] = r
colours[:, 1] = g
colours[:, 2] = b

sphere['colours'] = colours

plotter.add_mesh(
    sphere,
    scalars='colours',
    rgb=True,
    show_scalar_bar=False,
    smooth_shading=True
)


########
# Axes #
########

PlottingFunctions.add_axis(
    plotter,
    SPHERE_POSITION,
    [1, 0, 0],
    cylinder_length=AXIS_CYLINDER_LENGTH,
    cylinder_radius=AXIS_CYLINDER_RADIUS,
    cone_length=AXIS_CONE_LENGTH,
    cone_radius=AXIS_CONE_RADIUS
)

PlottingFunctions.add_axis(
    plotter,
    SPHERE_POSITION,
    [0, 1, 0],
    cylinder_length=AXIS_CYLINDER_LENGTH,
    cylinder_radius=AXIS_CYLINDER_RADIUS,
    cone_length=AXIS_CONE_LENGTH,
    cone_radius=AXIS_CONE_RADIUS
)

PlottingFunctions.add_axis(
    plotter,
    SPHERE_POSITION,
    [0, 0, 1],
    cylinder_length=AXIS_CYLINDER_LENGTH,
    cylinder_radius=AXIS_CYLINDER_RADIUS,
    cone_length=AXIS_CONE_LENGTH,
    cone_radius=AXIS_CONE_RADIUS
)

plotter.screenshot('/tmp/Preimages.png')
preimages_image = np.asarray(Image.open('/tmp/Preimages.png'))
axs[0, 0].imshow(preimages_image, interpolation='none')
axs[0, 0].axis('off')


#################
# xy-Axis Slice #
#################

single_slice_xy = data.slice(normal=[0, 0, 1])
single_slice_yz = data.slice(normal=[1, 0, 0])

m_slice = single_slice_xy.get_array('m').reshape((Nx, Ny, 3))

# Use my vectorised HLS to RGB function to get colours
rgb_array = PlottingFunctions.vec_to_RGB(m_slice)

im = axs[0, 1].imshow(rgb_array, origin='lower', interpolation='bilinear')
quiver_plot(axs[0, 1], m_slice, 0, 1, MAGNETISATION_STRIDE)

axs[0, 1].set_xlabel('$x$')
axs[0, 1].set_ylabel('$y$', labelpad=15)

axs[0, 1].set_xticks([])
axs[0, 1].set_yticks([])

axs[0, 1].spines['bottom'].set_color(XY_PLANE_OUTLINE_COLOUR)
axs[0, 1].spines['top'].set_color(XY_PLANE_OUTLINE_COLOUR)
axs[0, 1].spines['right'].set_color(XY_PLANE_OUTLINE_COLOUR)
axs[0, 1].spines['left'].set_color(XY_PLANE_OUTLINE_COLOUR)

axs[0, 1].spines['bottom'].set_linewidth(SPINES_LINEWIDTH)
axs[0, 1].spines['top'].set_linewidth(SPINES_LINEWIDTH)
axs[0, 1].spines['right'].set_linewidth(SPINES_LINEWIDTH)
axs[0, 1].spines['left'].set_linewidth(SPINES_LINEWIDTH)


#################
# yz-Axis Slice #
#################

m_slice = single_slice_yz.get_array('m').reshape((Ny, Nz, 3))

y = np.arange(Ny)
z = np.arange(Nz)
Y, Z = np.meshgrid(y, z)

# Use my vectorised HLS to RGB function to get colours
rgb_array = PlottingFunctions.vec_to_RGB(m_slice)

im = axs[0, 2].imshow(rgb_array, origin='lower', interpolation='bilinear')
quiver_plot(axs[0, 2], m_slice, 1, 2, MAGNETISATION_STRIDE)

axs[0, 2].set_xlabel('$y$')
axs[0, 2].set_ylabel('$z$', labelpad=15)

axs[0, 2].set_xticks([])
axs[0, 2].set_yticks([])

axs[0, 2].spines['bottom'].set_color(YZ_PLANE_OUTLINE_COLOUR)
axs[0, 2].spines['top'].set_color(YZ_PLANE_OUTLINE_COLOUR)
axs[0, 2].spines['right'].set_color(YZ_PLANE_OUTLINE_COLOUR)
axs[0, 2].spines['left'].set_color(YZ_PLANE_OUTLINE_COLOUR)

axs[0, 2].spines['bottom'].set_linewidth(SPINES_LINEWIDTH)
axs[0, 2].spines['top'].set_linewidth(SPINES_LINEWIDTH)
axs[0, 2].spines['right'].set_linewidth(SPINES_LINEWIDTH)
axs[0, 2].spines['left'].set_linewidth(SPINES_LINEWIDTH)


################################################
# Set Up 3D Emergent Magnetic Field Plot Scene #
################################################

plotter = pv.Plotter(window_size=WINDOW_SIZE, off_screen=True)
plotter.enable_anti_aliasing('ssaa')
data = pv.read(FILENAME_EMERGENT_FIELD)
Nx, Ny, Nz = data.dimensions

bounding_box = data.outline()
plotter.add_mesh(bounding_box, line_width=2, color='black')

plotter.camera_position = [
    CAMERA_POSITION,
    CAMERA_FOCAL_POINT,
    CAMERA_VIEW_UP
]


####################################
# Downsample and Plot Vector Field #
####################################

points = data.points
vectors = data['ext_emergentmagneticfield_solidangle']
points_structured = points.reshape((*data.dimensions, 3))
vectors_structured = vectors.reshape((*data.dimensions, 3))
points_structured_downsampled = points_structured[::EMERGENT_FIELD_STRIDE, ::EMERGENT_FIELD_STRIDE, ::EMERGENT_FIELD_STRIDE, :]
vectors_structured_downsampled = vectors_structured[::EMERGENT_FIELD_STRIDE, ::EMERGENT_FIELD_STRIDE, ::EMERGENT_FIELD_STRIDE, :]
points_downsampled = points_structured_downsampled.reshape(-1, 3)
vectors_downsampled = vectors_structured_downsampled.reshape(-1, 3)
downsampled_data = pv.StructuredGrid(points_downsampled)
downsampled_data['vectors'] = vectors_downsampled
glyphs = downsampled_data.glyph(orient='vectors', factor=0.01)
plotter.add_mesh(glyphs, color='black', show_scalar_bar=False)


##############################
# Plot Slices Through System #
##############################

field = data.get_array('ext_emergentmagneticfield_solidangle').reshape(Nx, Ny, Nz, 3)
magnitudes_structured = np.sqrt(np.einsum('ijkl,ijkl->ijk', field, field))
magnitudes = magnitudes_structured.reshape(-1, 1)
data['magnitudes'] = magnitudes

single_slice_xy = data.slice(normal=[0, 0, 1])
single_slice_yz = data.slice(normal=[1, 0, 0])

field_slice_xy  = single_slice_xy.get_array('ext_emergentmagneticfield_solidangle').reshape((Nx, Ny, 3))
field_slice_yz  = single_slice_yz.get_array('ext_emergentmagneticfield_solidangle').reshape((Ny, Nz, 3))

magnitudes_xy = np.sqrt(np.einsum('ijk,ijk->ij', field_slice_xy, field_slice_xy))
magnitudes_yz = np.sqrt(np.einsum('ijk,ijk->ij', field_slice_yz, field_slice_yz))

single_slice_xy['magnitudes_xy'] = magnitudes_xy.reshape(-1, 1)
single_slice_yz['magnitudes_yz'] = magnitudes_yz.reshape(-1, 1)

plotter.add_mesh(
    single_slice_xy,
    scalars         = 'magnitudes_xy',
    clim            = [0., np.max(magnitudes)],
    opacity         = 0.5,
    cmap            = colour_map,
    show_scalar_bar = False
)

# Add outline for slice
outline = single_slice_xy.outline()
plotter.add_mesh(outline, line_width=SLICE_OUTLINE_WIDTH, color=XY_PLANE_OUTLINE_COLOUR)

plotter.add_mesh(
    single_slice_yz,
    scalars         = 'magnitudes_yz',
    clim            = [0., np.max(magnitudes)],
    opacity         = 0.5,
    cmap            = colour_map,
    show_scalar_bar = False
)

# Add outline for slice
outline = single_slice_yz.outline()
plotter.add_mesh(outline, line_width=SLICE_OUTLINE_WIDTH, color=YZ_PLANE_OUTLINE_COLOUR)


##########################################
# Plot Isosurfaces of Constant Magnitude #
##########################################

isosurfaces = [0.25 * np.max(magnitudes), 0.75 * np.max(magnitudes)]
contour = data.contour(isosurfaces=isosurfaces, scalars='magnitudes')
plotter.add_mesh(
    contour,
    smooth_shading  = True,
    cmap            = colour_map,
    opacity         = 0.25,
    clim            = [0., np.max(magnitudes)],
    show_scalar_bar = False
)

plotter.screenshot('/tmp/EmergentField.png')
emergent_field_image = np.asarray(Image.open('/tmp/EmergentField.png'))
axs[1, 0].imshow(emergent_field_image, interpolation='none')
axs[1, 0].axis('off')


###############################
# Emergent Field Quiver Plots #
###############################

single_slice_xy = data.slice(normal=[0, 0, 1])
field_slice_xy  = single_slice_xy.get_array('ext_emergentmagneticfield_solidangle').reshape((Nx, Ny, 3))
quiver_plot(axs[1, 1], field_slice_xy, 0, 1, EMERGENT_FIELD_STRIDE_QUIVER)
axs[1, 1].set_xlabel('$x$')
axs[1, 1].set_ylabel('$y$', labelpad=15)

axs[1, 1].set_xticks([])
axs[1, 1].set_yticks([])

axs[1, 1].spines['bottom'].set_color(XY_PLANE_OUTLINE_COLOUR)
axs[1, 1].spines['top'].set_color(XY_PLANE_OUTLINE_COLOUR)
axs[1, 1].spines['right'].set_color(XY_PLANE_OUTLINE_COLOUR)
axs[1, 1].spines['left'].set_color(XY_PLANE_OUTLINE_COLOUR)

axs[1, 1].spines['bottom'].set_linewidth(SPINES_LINEWIDTH)
axs[1, 1].spines['top'].set_linewidth(SPINES_LINEWIDTH)
axs[1, 1].spines['right'].set_linewidth(SPINES_LINEWIDTH)
axs[1, 1].spines['left'].set_linewidth(SPINES_LINEWIDTH)


single_slice_yz = data.slice(normal=[1, 0, 0])
field_slice_yz  = single_slice_yz.get_array('ext_emergentmagneticfield_solidangle').reshape((Ny, Nz, 3))
quiver_plot(axs[1, 2], field_slice_yz, 1, 2, EMERGENT_FIELD_STRIDE_QUIVER)
axs[1, 2].set_xlabel('$y$')
axs[1, 2].set_ylabel('$z$', labelpad=15)

axs[1, 2].set_xticks([])
axs[1, 2].set_yticks([])

axs[1, 2].spines['bottom'].set_color(YZ_PLANE_OUTLINE_COLOUR)
axs[1, 2].spines['top'].set_color(YZ_PLANE_OUTLINE_COLOUR)
axs[1, 2].spines['right'].set_color(YZ_PLANE_OUTLINE_COLOUR)
axs[1, 2].spines['left'].set_color(YZ_PLANE_OUTLINE_COLOUR)

axs[1, 2].spines['bottom'].set_linewidth(SPINES_LINEWIDTH)
axs[1, 2].spines['top'].set_linewidth(SPINES_LINEWIDTH)
axs[1, 2].spines['right'].set_linewidth(SPINES_LINEWIDTH)
axs[1, 2].spines['left'].set_linewidth(SPINES_LINEWIDTH)


###############################
# Emergent Field Colour Plots #
###############################

magnitudes_xy = np.sqrt(np.einsum('ijk,ijk->ij', field_slice_xy, field_slice_xy))
magnitudes_yz = np.sqrt(np.einsum('ijk,ijk->ij', field_slice_yz, field_slice_yz))

axs[1, 1].imshow(magnitudes_xy,
    interpolation='bilinear',
    cmap=colour_map,
    extent=[0, Nx-1, 0, Ny-1],
    vmin=0.,
    vmax=np.maximum(np.max(magnitudes_xy), np.max(magnitudes_yz))
)

im = axs[1, 2].imshow(magnitudes_yz,
    interpolation='bilinear',
    cmap=colour_map,
    extent=[0, Nx-1, 0, Ny-1],
    vmin=0.,
    vmax=np.maximum(np.max(magnitudes_xy), np.max(magnitudes_yz))
)

ax_cbar = inset_axes(axs[1, 2], width='5%', height='100%', loc='lower left',
    bbox_to_anchor = (1.04, 0., 1, 1), bbox_transform = axs[1, 2].transAxes,
    borderpad = 0)

cb = fig.colorbar(im, cax=ax_cbar)
cb.ax.set_ylabel(r'$|\boldsymbol{F}|$', rotation=0, loc='top')


##########$##################################
# Mark Boundaries of Compact Support Region #
##########$##################################

with open('Sim.mx3') as f:
    pattern = r'HopfionCompactSupport\(([\d.]+),\s*([\d.]+)\)'
    content = f.read()
    major_radius, minor_radius = re.search(pattern, content).groups()

major_radius = float(major_radius)
minor_radius = float(minor_radius)


def draw_circle(ax, circle_radius, horizontal_position):

    """ Draw circles to show where the compact support region is. """

    circle = Circle(
        (horizontal_position, 0.5),
        circle_radius,
        edgecolor=COMPACT_SUPPORT_COLOUR,
        facecolor='none',
        linewidth=2,
        linestyle=COMPACT_SUPPORT_STYLE,
        transform=ax.transAxes
    )
    ax.add_patch(circle)

# Circles in x-plane
draw_circle(axs[0, 1], major_radius - minor_radius, 0.5)
draw_circle(axs[0, 1], major_radius + minor_radius, 0.5)
draw_circle(axs[1, 1], major_radius - minor_radius, 0.5)
draw_circle(axs[1, 1], major_radius + minor_radius, 0.5)

# Circles in yz-plane
draw_circle(axs[0, 2], minor_radius, 0.5 - major_radius)
draw_circle(axs[0, 2], minor_radius, 0.5 + major_radius)
draw_circle(axs[1, 2], minor_radius, 0.5 - major_radius)
draw_circle(axs[1, 2], minor_radius, 0.5 + major_radius)


plt.tight_layout()
plt.subplots_adjust(wspace=-0.1)
plt.savefig('HopfionFigure.pdf', format='pdf')
