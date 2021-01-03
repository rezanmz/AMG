from AMG.multigrid import Multigrid
import AMG
import pyamg
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import norm

A = pyamg.gallery.poisson((50, 50), format='csr')
b = np.random.rand(A.shape[0])
threshold = 1e-5

AMG.init()
AMG.pre_smooth_iters = 10
AMG.post_smooth_iters = 10
beck_norm = [10]
solution = None
while beck_norm[-1] > threshold:
    ml = Multigrid(
        adjacency_matrix=A,
        right_hand_side=b,
        coarsening_method='beck',
        smallest_coarse_size=10,
        initial_x=solution
    )
    solution = ml.v_cycle()
    del ml
    beck_norm.append(norm(b - A.dot(solution)))
beck_norm = beck_norm[1:]

AMG.init()
AMG.pre_smooth_iters = 10
AMG.post_smooth_iters = 10
sa_norm = [10]
solution = None
while sa_norm[-1] > threshold:
    ml = Multigrid(
        adjacency_matrix=A,
        right_hand_side=b,
        coarsening_method='standard-aggregation',
        smallest_coarse_size=10,
        initial_x=solution
    )
    solution = ml.v_cycle()
    del ml
    sa_norm.append(norm(b - A.dot(solution)))
sa_norm = sa_norm[1:]

AMG.init()
AMG.pre_smooth_iters = 10
AMG.post_smooth_iters = 10
gl_coarsener_norm = [1]
solution = None
while gl_coarsener_norm[-1] > threshold:
    ml = Multigrid(
        adjacency_matrix=A,
        right_hand_side=b,
        coarsening_method='gl-coarsener',
        smallest_coarse_size=10,
        initial_x=solution
    )
    solution = ml.v_cycle()
    del ml
    gl_coarsener_norm.append(norm(b - A.dot(solution)))
gl_coarsener_norm = gl_coarsener_norm[1:]

plt.plot(range(len(gl_coarsener_norm)),
         gl_coarsener_norm, label='GL-Coarsener')
plt.plot(range(len(sa_norm)),
         sa_norm, label='Standard Aggregation')
plt.plot(range(len(beck_norm)), beck_norm, label='Beck')
plt.legend()
plt.show()
