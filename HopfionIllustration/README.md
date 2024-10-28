# Hopfion Figure

To reproduce,

1. Run the MuMax3 script: `mumax3 -o Data Sim.mx3`.
2. Convert the OVF files to VTK files: `cd Data && mumax3-convert -vtk binary *.ovf && cd ..`
3. Run the Python script: `python HopfionFigure.py`

