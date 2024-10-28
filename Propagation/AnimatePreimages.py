from colorsys import hls_to_rgb
import sys
import os
import numpy as np
import pyvista as pv
import pandas as pd

sys.path.append('..')
from Utils import PlottingFunctions, QuantityReader


pv.global_theme.allow_empty_mesh = True


#########################
# Constant Declarations #
#########################

SPHERE_POSITION      = (10, 0, 0)
SPHERE_RADIUS        = 3
CONE_LENGTH          = 2
CONE_RADIUS          = 1
WINDOW_SIZE          = [1000, 1000]
CAMERA_POSITION      = (35, 35, 30)
CAMERA_FOCAL_POINT   = (10, 10, 13)
CAMERA_VIEW_UP       = (0, 0, 1)
SIMULATION_DIRECTORY = 'Data'
STILLS_DIRECTORY     = 'Stills'
VIDEOS_DIRECTORY     = 'Videos'
PARAMS_FILE          = 'params'


################################
# Collect Files to Be Rendered #
################################

simulation_files = []

for simulation_file in os.listdir(SIMULATION_DIRECTORY):
    if simulation_file.startswith('m') and simulation_file.endswith('.vts'):
        simulation_files.append(simulation_file)

simulation_files = sorted(simulation_files)


################
# Render Files #
################

os.makedirs(STILLS_DIRECTORY, exist_ok=True)

for simulation_idx, simulation_file in enumerate(simulation_files):

    print(simulation_file, end='\r')


    ################
    # Set Up Scene #
    ################

    plotter = pv.Plotter(window_size=WINDOW_SIZE, off_screen=True)
    plotter.enable_anti_aliasing('ssaa')

    plotter.camera_position = [
        CAMERA_POSITION,
        CAMERA_FOCAL_POINT,
        CAMERA_VIEW_UP
    ]


    ####################
    # Render Preimages #
    ####################

    data = pv.read(SIMULATION_DIRECTORY + '/' + simulation_file)

    # Scale according to natural length units
    A = QuantityReader.read_quantity('A', PARAMS_FILE)
    K = QuantityReader.read_quantity('K', PARAMS_FILE)
    length_units = np.sqrt(A / K)
    data.points /= length_units

    preimage_azimuthal_angles = np.linspace(0, 2*np.pi, 12)

    for i, angle in enumerate(preimage_azimuthal_angles):

        PlottingFunctions.plot_preimage(
            data,
            plotter,
            (np.cos(angle), np.sin(angle), 0),
            preimage_closeness = 0.97
        )


    #######################
    # Render Bounding Box #
    #######################

    bounding_box = data.outline()
    plotter.add_mesh(bounding_box, line_width=2, color='black')


    ###################
    # Save Screenshot #
    ###################

    plotter.screenshot(STILLS_DIRECTORY + f'/{simulation_idx:06}.png')
