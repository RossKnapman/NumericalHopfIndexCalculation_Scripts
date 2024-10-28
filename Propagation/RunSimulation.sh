#!/bin/bash

source params

sed "s/{{ A }}/$A/" Sim.mx3.template |\
sed "s/{{ K }}/$K/" |\
sed "s/{{ Ms }}/$Ms/" |\
sed "s/{{ Lxy }}/$Lxy/" |\
sed "s/{{ Nxy }}/$Nxy/" |\
sed "s/{{ Nz }}/$Nz/" |\
sed "s/{{ N_uniform }}/$N_uniform/" |\
sed "s/{{ OutputTimestep }}/$OutputTimestep/" |\
sed "s/{{ SimulationTime }}/$SimulationTime/" |\
sed "s/{{ Alpha }}/$Alpha/" > Sim.mx3

mumax3 -f -o Data Sim.mx3
cd Data
mumax3-convert -vtk binary *.ovf
cd ..
