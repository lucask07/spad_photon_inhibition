""" Named hand-crafted spatial kernels

Named heuristic kernels as a dictionary that can be loaded and used
The tuple has two elements: 0: is the kernel matrix, 1: is the color for plotting

Use by:
from kernels import spatial_kernels

2023/5/24
Lucas J. Koerner, koerner.lucas@stthomas.edu
"""

import numpy as np

spatial_kernels = {'optimal': (np.array([[1,1,1], 
                                        [1,7.7,1],
                                        [1,1,1]]), 'k'),
                    'flip_laplacian': (np.array([[1,1,1], 
                                        [1,8,1],
                                        [1,1,1]]), 'k'),
                    'poor_gradient': (np.array([[0.7,1.5,0.7], 
                                        [1.5,-5.4,1.5],
                                        [0.7,1.5,0.7]]), 'm'),
                    'laplacian': (np.array([[-1,-1,-1], 
                                            [-1, 8,-1],
                                            [-1,-1,-1]]), 'm'),
                    'laplacian_and_avg': ([np.array([[-1,-1,-1], 
                                            [-1, 8,-1],
                                            [-1,-1,-1]]),  
                                            np.array([[1,1,1], 
                                                   [1,1,1],
                                                    [1,1,1]])], # list of two numpy arrays 
                                            'm', ['and']), # combine operator 
                    'laplacian_and_avg_or_avg': ([np.array([[-1,-1,-1], 
                                            [-1, 8,-1],
                                            [-1,-1,-1]]),  
                                            np.array([[1,1,1], 
                                                   [1,1,1],
                                                    [1,1,1]]),
                                            np.array([[1,1,1], 
                                                   [1,1,1],
                                                    [1,1,1]])], # list of three numpy arrays 
                                            'm', ['and','or']), # combine operators 
                    'good_mse': (np.array([[0.4,-1.9,0.4], 
                                        [-1.9,5.7,-1.9],
                                        [0.4,-1.9,0.4]]), 'y'),
                    'limited_mask': (np.array([[-1.0,-2.3,-1.0], 
                                        [-1.3,-0.2,-2.3],
                                        [-1.0,-2.3,-1.0]]), 'g'),
                    'red_cluster': (np.array([[1.4,1.3,1.4], 
                                        [1.3,6.3,1.3],
                                        [1.4,1.3,1.4]]), 'r'),
                    'poormse_goodmask': (np.array([[1.5,1.4,1.5], 
                                        [1.4,-3.9,1.4],
                                        [1.5,1.4,1.5]]), 'c'),
                    'edge_h1':          (np.array([[1,-1,1], 
                                                [-1,2,-1],
                                                [1,-1,1]]), 'greenyellow'),
                    'edge_h2':          (np.array([[1,1,1], 
                                                   [-1,2,-1],
                                                    [1,1,1]]), 'tab:brown'),
                    'neighbor8':          (np.array([[1,1,1], 
                                                   [1,1,1],
                                                    [1,1,1]]), 'k'),
                    'neighbor4':          (np.array([[0,1,0], 
                                                   [1,1,1],
                                                    [0,1,0]]), 'm'),
                    'cross':              (np.array([[0,1,0], 
                                                     [1,0,1],
                                                     [0,1,0]]), 'm'),
                'single_pix_bright':          (np.array([[0,0,0], 
                                                   [0,1,0],
                                                    [0,0,0]]), 'r'),
                    'single_pix_dark':          (np.array([[0,0,0], 
                                                   [0,-1,0],
                                                    [0,0,0]]), 'k'),
                    'neighbor8_neg':          (np.array([[-1,-1,-1], 
                                                   [-1,-1,-1],
                                                    [-1,-1,-1]]), 'k'),
'large5x5':          (np.array([[1,1,1,1,1], 
                                                   [1,1,1,1,1],
                                                   [1,1,1,1,1],
                                                   [1,1,1,1,1],
                                                    [1,1,1,1,1]]), 'tab:purple'),
'large7x7':          (np.ones((7,7)), 'tab:purple'),
'large13x13':          (np.ones((13,13)), 'tab:purple'),
'large19x19':          (np.ones((19,19)), 'tab:purple'),
'all_ones':          ('all_ones', 'tab:purple'),
                    'no_v_discrepancy':  ('no_v_discrepancy', 'tab:pink'),
                    'no_nophotons':  ('no_nophotons', 'tab:olive')

                    }

