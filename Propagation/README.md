# Propagating Hopfion

To reproduce the manuscript figure:

1. Run the simulation by running the script `RunSimulation.sh`
2. Run the Python script: `python PropagationFigure.py`


To generate the supplementary videos, execute the Python scripts `AnimatePreimages.py` and `AnimateSliceHopfIndex.py`. For the preimages, the individual frame are output to a new directory `Stills`. You can put them together into a video using e.g. `ffmpeg -y -framerate 25 -i Stills/%06d.png -r 25 -pix_fmt yuv420p Preimages.mp4`.
