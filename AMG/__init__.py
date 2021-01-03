def init():
    global prolongators, pre_smoother, post_smoother, pre_smooth_iters, post_smooth_iters
    pre_smoother = 'jacobi'
    pre_smooth_iters = 2
    post_smoother = 'jacobi'
    post_smooth_iters = 2
    prolongators = {}
