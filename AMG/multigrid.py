from AMG.coarsener import Coarsener
import AMG
from .smoother import Smoother
from scipy.sparse import linalg
import numpy as np


class Multigrid:
    _total_levels = 0

    # Constructor
    def __init__(
        self,
        adjacency_matrix,
        right_hand_side,
        coarsening_method,
        smallest_coarse_size,
        initial_x=None,
    ):
        # With each creation of a Multigrid instance, _total_levels is
        # increased in the constructor of the class.
        Multigrid._total_levels += 1
        # We store the current level number in 'level' attribute of the object
        # As we go deeper in the cycle, the we have a higher 'level'
        self.level = Multigrid._total_levels
        # In a basic multigrid solver, the initial solution and
        # the adjacency matrix will be passed to multigrid
        # cycle from a finer level
        self.right_hand_side = right_hand_side
        self.number_of_equations = len(right_hand_side)
        self.adjacency_matrix = adjacency_matrix
        self.coarsening_method = coarsening_method
        self.smallest_coarse_size = smallest_coarse_size
        self.initial_x = initial_x
        if self.initial_x is None:
            self.initial_x = np.zeros(adjacency_matrix.shape[0])

    # Destructor
    def __del__(self):
        """
        In the destructor, as we are done with the current corse level,
        we will go to a finer level and decrease the total level by 1
        """
        Multigrid._total_levels -= 1

    def v_cycle(self):
        print('Current level:', self.level, end='\t')
        print('Number of unknowns:', self.number_of_equations)
        # Pre-Smoothing
        smoother = Smoother(
            A=self.adjacency_matrix,
            b=self.right_hand_side,
            n=AMG.pre_smooth_iters,
            x=self.initial_x,
        )
        if AMG.pre_smoother == 'jacobi':
            v, _ = smoother.jacobi()
        elif AMG.pre_smoother == 'gauss_seidel':
            v, _ = smoother.gauss_seidel()

        # Calculate the residual --> Original system=>Ax=b   Residual=>b-Av=r
        residual = self.right_hand_side - self.adjacency_matrix.dot(v)

        # Check if prolongator is already provided in input
        # To speedup the process, we save the calculated prolongator in the previous iterations for future use
        if self.level not in AMG.prolongators:
            coarsener = Coarsener(self.adjacency_matrix)
            if self.coarsening_method == 'beck':
                prolongation_operator = coarsener.beck()
            elif self.coarsening_method == 'gl-coarsener':
                prolongation_operator = coarsener.gl_coarsener()
            elif self.coarsening_method == 'standard-aggregation':
                prolongation_operator = coarsener.standard_aggregation(
                    0.25)
            AMG.prolongators[self.level] = prolongation_operator
        else:
            prolongation_operator = AMG.prolongators[self.level]
        # Restriction operator is the transpose of prolongation operator
        restriction_operator = prolongation_operator.T
        restricted_adjacency = restriction_operator.dot(
            self.adjacency_matrix).dot(prolongation_operator)

        restricted_residual = restriction_operator.dot(residual)

        # If coarse_number_of_equations <= self.smallest_coarse_size, solve the coarse correction directly.
        if len(restricted_residual) <= self.smallest_coarse_size:
            coarse_correction = linalg.spsolve(
                restricted_adjacency, restricted_residual)

        # If we haven't reached the desired depth(level) in V-cycle,
        # perform another multigrid V-cycle to find the coarse correction.
        else:
            multigrid = Multigrid(
                adjacency_matrix=restricted_adjacency,
                right_hand_side=restricted_residual,
                coarsening_method=self.coarsening_method,
                smallest_coarse_size=self.smallest_coarse_size
            )
            coarse_correction = multigrid.v_cycle()

        # Finally, interpolate coarse correction to the fine grid using
        # the prolongator operator ---> P.CoarseCorrection = FineCorrection
        fine_correction = prolongation_operator.dot(coarse_correction)

        # Post-Smoothing
        smoother = Smoother(
            A=self.adjacency_matrix,
            b=self.right_hand_side,
            n=AMG.post_smooth_iters,
            x=fine_correction + self.initial_x,
        )
        if AMG.post_smoother == 'jacobi':
            v, _ = smoother.jacobi()
        elif AMG.post_smoother == 'gauss_seidel':
            v, _ = smoother.gauss_seidel()

        return v
