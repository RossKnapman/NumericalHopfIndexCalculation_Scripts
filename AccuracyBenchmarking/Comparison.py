import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from colorsys import hls_to_rgb
import os
from PIL import Image

pv.global_theme.allow_empty_mesh = True

# Matplotlib stylistic things
plt.style.use('seaborn-v0_8-colorblind')
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['axes.labelsize'] = 18

df                          = pd.read_csv('Data/table.txt', delimiter='\t')
N                           = df['N ()'].to_numpy()
H_finitedifferencefivepoint = df['ext_hopfindex_fivepointstencil ()']
H_finitedifferencetwopoint  = df['ext_hopfindex_twopointstencil ()']
H_solidangle                = df['ext_hopfindex_solidangle ()']
H_fourier                   = df['ext_hopfindex_solidanglefourier ()']

df_fem = pd.read_csv('FEMresults.dat', delimiter='\t')
Delta_fem = df_fem['cell_size']
H_fem = df_fem['hopf_idx']

fig, ax = plt.subplots()

Delta = 1. / N

ax.plot(Delta, H_finitedifferencefivepoint, linestyle='solid', marker='.', label='ext_hopfindex_fivepointstencil')
ax.plot(Delta, H_finitedifferencetwopoint, linestyle='solid', marker='.', label='ext_hopfindex_twopointstencil')
ax.plot(Delta, H_solidangle, linestyle='solid', marker='.', label='ext_hopfindex_solidangle')
ax.plot(Delta, H_fourier, linestyle='solid', marker='.', label='ext_hopfindex_solidanglefourier')
ax.plot(Delta_fem, H_fem, linestyle='solid', marker='.', label='FEM')
ax.axhline(1.0, color='black', linestyle='dashed', alpha=0.5)
ax.set_xlabel(r'$\Delta$')
ax.set_ylabel(r'Calculated $H$')

ax_ins = ax.inset_axes(
    [0.65, 0.1, 0.3, 0.4],
    xlim=(0.02, 0.01),
    ylim=(0.995, 1.005)
)

ax_ins.plot(Delta, H_finitedifferencefivepoint, linestyle='solid', marker='.', label='ext_hopfindex_fivepointstencil')
ax_ins.plot(Delta, H_finitedifferencetwopoint, linestyle='solid',  marker='.', label='ext_hopfindex_twopointstencil')
ax_ins.plot(Delta, H_solidangle, linestyle='solid', marker='.', label='ext_hopfindex_solidangle')
ax_ins.plot(Delta, H_fourier, linestyle='solid', marker='.', label='ext_hopfindex_solidanglefourier')
ax_ins.plot(Delta_fem, H_fem, linestyle='solid', marker='.', label='FEM')

ax_ins.axhline(1.0, color='black', linestyle='dashed', alpha=0.5)

ax.indicate_inset_zoom(ax_ins)
ax.invert_xaxis()

plt.legend(loc='lower left')

plt.savefig('Comparison.pdf', format='pdf')