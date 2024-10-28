import random
import numpy as np
import subprocess
import re
import matplotlib
import matplotlib.pyplot as plt
import pickle

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['axes.labelsize'] = 18
plt.style.use('seaborn-v0_8-colorblind')

MUMAX_BINARY     = '/path/to/mumax3'
NO_RUNS          = 100
GRID_SIDE_LENGTH = [i for i in range(20, 110, 10)]
methods          = [
    'ext_hopfindex_fivepointstencil',
    'ext_hopfindex_twopointstencil',
    'ext_hopfindex_solidangle',
    'ext_hopfindex_solidanglefourier'
]
Z_SCORE_TOLERANCE = 3.  # Consider data with more than this many standard deviations from the mean as outliers
results_arrays    = {method: np.zeros((NO_RUNS, len(GRID_SIDE_LENGTH)), dtype=float) for method in methods}


for run_idx in range(NO_RUNS):
    print(run_idx, end='\r')
    
    grid_side_length_shuffled = GRID_SIDE_LENGTH[:]  # New list to which changes do not alter original
    random.shuffle(grid_side_length_shuffled)  # Side lengths in a random order to reduce bias

    for sidelength_idx, sidelength in enumerate(grid_side_length_shuffled):

        methods_shuffled = methods[:]
        random.shuffle(methods_shuffled)  # Methods in a random order to reduce bias

        for method in methods_shuffled:

            # Run MuMax scripts, read out time
            with open('Sim.mx3.template', 'r') as template_file:
    
                content = template_file.read()
                content = content.replace('{{ N }}', str(sidelength))
                content = content.replace('{{ Method }}', method)

                with open('Sim.mx3', 'w') as simulation_file:
                    simulation_file.write(content)

            command     = [MUMAX_BINARY, '-f', '-o', 'Data', 'Sim.mx3']
            result      = subprocess.run(command, capture_output=True, text=True)
            output      = result.stdout
            before_time = int(re.search(r'Before\s(\d+)', output).group(1))
            after_time  = int(re.search(r'After\s(\d+)', output).group(1))

            sidelength_idx_original = GRID_SIDE_LENGTH.index(sidelength)
            results_arrays[method][run_idx, sidelength_idx_original] = after_time - before_time


for method in methods:

    calculation_times                    = results_arrays[method] / 1e6  # Convert time in ns to ms
    calculation_times_mean               = np.mean(calculation_times, axis=0)
    calculation_times_standard_deviation = np.std(calculation_times, axis=0)

    # Filter out outliers by removing points more than Z_SCORE_TOLERANCE standard deviations from the mean
    z_scores          = np.abs((calculation_times - calculation_times_mean) / calculation_times_standard_deviation)
    non_outliers_mask = z_scores < Z_SCORE_TOLERANCE
    no_runs_filtered  = np.sum(non_outliers_mask, axis=0)

    calculation_times_filtered       = np.where(non_outliers_mask, calculation_times, np.nan)
    calculation_times_mean_filtered  = np.nanmean(calculation_times_filtered, axis=0)
    calculation_times_standard_error = np.nanstd(calculation_times_filtered, axis=0) / np.sqrt(no_runs_filtered)

    plt.errorbar(GRID_SIDE_LENGTH, calculation_times_mean_filtered,
        yerr=calculation_times_standard_error, linestyle='-', marker='.', label=method)

    plt.xlabel(r'$N$')
    plt.ylabel(r'Calculation Time (ms)')
    plt.legend()
    
    plt.savefig('Benchmarking.pdf', format='pdf')
