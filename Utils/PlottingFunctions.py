from colorsys import hls_to_rgb
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib import animation


def plot_preimage(data, plotter, preimage_direction, preimage_closeness=0.97):

    """
    Plots a surface surrounding the preimage of the specified direction in the
    texture.

    Args:
        data (pyvista.DataSet): Dataset containing the magnetisation texture.
        plotter (pyvista.Plotter): Plotting object where everything is displayed.
        preimage_direction (ndarray): Unit vector corresponding to preimage direction.
        preimage_closeness: How close the isosurface is to the preimage (1 means
            infinitesimally close to the preimage line).

    """

    l = 0.5 * preimage_direction[2] + 0.5
    h = np.arctan2(preimage_direction[1], preimage_direction[0]) / (2*np.pi)
    r, g, b, = hls_to_rgb(h, l, 1.0)

    # We plot preimages by plotting the isosurface of (preimage_direction) · m = preimage_closeness
    preimage_direction = preimage_direction / np.linalg.norm(preimage_direction)
    preimagedirection_dot_m = np.dot(data['m'], preimage_direction)
    data['preimagedirection_dot_m'] = preimagedirection_dot_m
    contour = data.contour(isosurfaces=[preimage_closeness], scalars='preimagedirection_dot_m')
    plotter.add_mesh(contour, color=[r, g, b], smooth_shading=True)


def add_sphere_preimage(plotter, preimage_direction, sphere_position, sphere_radius,
    cone_length, cone_radius):

    """
    Adds a cone to the legend sphere corresponding to the preimage.

    Args:
        plotter (pyvista.Plotter): Plotting object where everything is displayed.
        preimage_direction (ndarray): Unit vector corresponding to preimage direction.
        sphere_position (tuple[float, float, float]): Position of the legend sphere.
        sphere_radius (float): Radius of legend sphere.
        cone_length (float): Length of cones indicating preimage directions.
        cone_radius (float): Radius of cones indicating preimage directions.

    """

    l = 0.5 * preimage_direction[2] + 0.5
    h = np.arctan2(preimage_direction[1], preimage_direction[0]) / (2*np.pi)
    r, g, b, = hls_to_rgb(h, l, 1.0)

    az   = np.arctan2(preimage_direction[1], preimage_direction[0])
    alt  = np.arccos(preimage_direction[2])
    cone = pv.Cone(
            center = (
                sphere_position[0] + sphere_radius * np.cos(az) * np.sin(alt),
                sphere_position[1] + sphere_radius * np.sin(az) * np.sin(alt),
                sphere_position[2] + sphere_radius * np.cos(alt)
            ),
            direction = (
                np.cos(az) * np.sin(alt),
                np.sin(az) * np.sin(alt),
                np.cos(alt)
            ),
            height=cone_length,
            radius=cone_radius,
            resolution=20)
    plotter.add_mesh(cone, color=[r, g, b], smooth_shading=True)


def add_axis(plotter, position, direction, cylinder_length=3, cylinder_radius=0.1, cone_length=0.5, cone_radius=0.25):

    """
    Adds an arrow to represent coordinate axes.

    Args:
        plotter (pyvista.Plotter): Plotting object where everything is displayed.
        position (tuple[float, float, float]): Position of the origin of the axes.
        direction (tuple[float, float, float]): Direction in which arrow points.

    """

    az  = np.arctan2(direction[1], direction[0])
    alt = np.arccos(direction[2])

    cylinder = pv.Cylinder(
        center = (
            position[0] + (cylinder_length/2) * np.cos(az) * np.sin(alt),
            position[1] + (cylinder_length/2) * np.sin(az) * np.sin(alt),
            position[2] + (cylinder_length/2) * np.cos(alt)
        ),
        direction = direction,
        height    = cylinder_length,
        radius    = cylinder_radius
    )

    cone = pv.Cone(
        center = (
            position[0] + (cylinder_length + cone_length/2) * np.cos(az) * np.sin(alt),
            position[1] + (cylinder_length + cone_length/2) * np.sin(az) * np.sin(alt),
            position[2] + (cylinder_length + cone_length/2) * np.cos(alt)
        ),
        direction = direction,
        height    = cone_length,
        radius    = cone_radius,
        resolution = 20
    )

    plotter.add_mesh(cylinder, color='black')
    plotter.add_mesh(cone, color='black')


def animate_quantity(times_array, quantity_array, out_name, ylabel):

    """
    Animate e.g. Hopf index or magnetic field z-component.

    Args:
        times_array (ndarray): Array of times.
        quantity_array (ndarray): Array of the quantity to be animated.
        out_name (str): Name of the video file to be output.
        ylabel (str): Name of the y-axis label.

    """

    fig, ax = plt.subplots()

    plot, = ax.plot(times_array[0], quantity_array[0], animated=True)

    ax.set_ylim(0.9*np.min(quantity_array), 1.1*np.max(quantity_array))
    ax.set_xlim(0, np.max(times_array))
    ax.set_ylabel(ylabel)

    anim = animation.FuncAnimation(
        fig,
        lambda frame: plot.set_data(times_array[:frame+1], quantity_array[:frame+1]),
        iter(range(len(times_array))),
        save_count=len(times_array)
    )

    anim.save(out_name, fps=25, writer='ffmpeg')


def vec_to_RGB(m):

    """
    Allows for HLS colour plot of magnetization without having to manually loop pixel-by-pixel.
    Essentially a vectorised version of colorsys.hls_to_rgb.

    Args:
        m (ndarray): Array of shape `(Nx, Ny, 3)`.

    Returns:
        An array of shape `(Nx, Ny, 3)`, where the final axis is contains the [R, G, B] values.
    
    """

    m[np.where(m == -0.)] = 0.  # Change -0. to 0.

    ONE_THIRD = 1.0/3.0
    ONE_SIXTH = 1.0/6.0
    TWO_THIRD = 2.0/3

    def _v(m1, m2, hue):

        hue = hue % 1.

        outValue = np.zeros(m1.shape, dtype=float)

        whereHueLessOneSixth = np.where(hue < ONE_SIXTH)
        outValue[whereHueLessOneSixth] = m1[whereHueLessOneSixth] + \
            (m2[whereHueLessOneSixth] - m1[whereHueLessOneSixth]) * \
            hue[whereHueLessOneSixth] * 6.

        whereHueLessHalf = np.where((hue < 0.5) & (hue >= ONE_SIXTH))
        outValue[whereHueLessHalf] = m2[whereHueLessHalf]

        whereHueLessTwoThird = np.where((hue < TWO_THIRD) & (hue >= 0.5))
        outValue[whereHueLessTwoThird] = m1[whereHueLessTwoThird] + \
            (m2[whereHueLessTwoThird] - m1[whereHueLessTwoThird]) * \
            (TWO_THIRD - hue[whereHueLessTwoThird]) * 6.0

        remainingPositions = np.where(hue >= TWO_THIRD)
        outValue[remainingPositions] = m1[remainingPositions]

        return outValue

    s = np.linalg.norm(m, axis=2)
    l = 0.5 * m[:, :, 2] + 0.5
    h = np.arctan2(m[:, :, 1], m[:, :, 0]) / (2 * np.pi)

    h[np.where(h > 1)] -= 1
    h[np.where(h < 0)] += 1

    rgbArray = np.zeros(m.shape, dtype=float)

    wheresIsZero = np.where(s == 0.)

    try:
        rgbArray[wheresIsZero] = np.array(
            [float(l[wheresIsZero]), float(l[wheresIsZero]), float(l[wheresIsZero])])

    except TypeError:  # No such points found
        pass

    m2 = np.zeros((rgbArray.shape[0], rgbArray.shape[1]), dtype=float)

    wherelIsLessThanHalf = np.where(l <= 0.5)
    m2[wherelIsLessThanHalf] = l[wherelIsLessThanHalf] * \
        (1.0 + s[wherelIsLessThanHalf])

    wherelIsMoreThanHalf = np.where(l > 0.5)
    m2[wherelIsMoreThanHalf] = l[wherelIsMoreThanHalf] + \
        s[wherelIsMoreThanHalf] - l[wherelIsMoreThanHalf] * s[wherelIsMoreThanHalf]

    m1 = 2.0 * l - m2

    rgbArray[:, :, 0] = _v(m1, m2, h+ONE_THIRD)
    rgbArray[:, :, 1] = _v(m1, m2, h)
    rgbArray[:, :, 2] = _v(m1, m2, h-ONE_THIRD)

    return rgbArray
