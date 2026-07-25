# Method overview: reading the CE-PINN code

This document walks through the four source files that make up the CE-PINN library. It is written for a reader who wants to understand the full method from code alone, without jumping between files and guessing what each piece does. The paper describes the mathematics; this guide describes the implementation.


## The big picture

The library lives in `ce_pinn/` and contains four files. The file `equations.py` holds exact solutions and conservation law evaluators for each PDE. The file `network.py` defines the neural network and the hard boundary condition layers. The file `projection.py` implements the two projection strategies (polynomial and Helmholtz) that force the network output to satisfy conservation laws exactly. The file `train.py` ties them together into a training loop with logging and checkpointing.

Each experiment is a standalone Jupyter notebook in `notebooks/`. A notebook sets the hyperparameters, builds the model, trains it, evaluates it, and (if the run succeeds) pushes the results to GitHub. The library only supplies the building blocks.


## equations.py

This file provides reference solutions in two forms: NumPy functions for validation, and PyTorch functions for use inside the computational graph during training.

For the KdV equation, the one-soliton solution is the standard sech-squared profile with speed proportional to the square of the wavenumber. The two-soliton solution uses Hirota's bilinear method. The implementation computes the Hirota tau-function via log-sum-exp to avoid floating-point overflow when the exponentials grow large. Spatial derivatives of the tau-function are then computed with NumPy's gradient function to recover the physical solution.

For the RLW equation, the single soliton has amplitude 3c and propagation speed 1 + epsilon times c. The two-soliton initial condition is a simple superposition of two well-separated single solitons (the RLW equation does not admit a clean two-soliton formula, so no exact solution is available for later times). The undular bore initial condition is a smoothed step function, where the parameter d controls the width of the transition region.

For the KP equation, the lump solution is an algebraically decaying 2D structure that moves at speed c = negative sigma. The line soliton extends the 1D KdV sech-squared profile into two dimensions with an additional transverse wavenumber.

At the bottom of the file, `compute_invariants_1d` evaluates the three conservation integrals for a given solution snapshot using PyTorch's built-in trapezoidal rule. It automatically selects between KdV-style and RLW-style formulas depending on whether the dispersion parameter mu is supplied.


## network.py

The MLP class is a standard feed-forward network: a sequence of linear layers with SiLU activations and a final linear output. All weights are initialised with Kaiming uniform and all biases start at zero. The network uses float64 throughout.

The interesting part is the hard boundary condition embedding. The class `HardBC1D` wraps the raw MLP and replaces its output with a constructed function that satisfies the initial and boundary conditions by design. The construction follows the Lagaris approach. The output is

    u_hat(x, t) = A(x, t) + B(x, t) * N(x, t; theta)

where A is a function that interpolates between the initial condition and the boundary values, and B is a product of factors that vanish at the boundaries and at t = 0. The term N is the raw MLP output. Because B is zero everywhere A is constrained, the network can only add corrections where the solution is free, and the boundary and initial conditions are satisfied exactly regardless of the network parameters.

For non-homogeneous boundary conditions (the undular bore, for instance, has different values at the left and right boundaries that vary in time), the constructor accepts optional functions `ga_fn` and `gb_fn` that supply the boundary values as functions of time.

The class `HardBC2D` does the same thing for the KP equation. The mask B is the product t * (x - xa) * (xb - x) * (y - ya) * (yb - y), which vanishes on all four spatial boundaries and at t = 0.


## projection.py

This file implements the two strategies for enforcing conservation laws as hard constraints on the network output, rather than as soft penalties in the loss function.

The polynomial projection is the simpler of the two. Given a network output u_tilde at a single time level, it applies an affine rescaling (shift and scale) to produce a corrected u_star whose first two integrals match the target invariants C1 and C2. This amounts to matching the mean and variance of u_star to the values implied by the targets. The operation is differentiable and has negligible cost. It works well for invariants that depend only on u, such as the mass integral u dx and the momentum integral u-squared dx for KdV. This is the baseline, and it reproduces the approach used by the SDIFP method in the literature.

The Helmholtz projection is the main contribution. It handles invariants that depend on spatial derivatives of u, such as the RLW momentum integral u-squared + mu times u_x-squared. An affine rescaling cannot preserve such an invariant because rescaling u changes u_x in a different way than it changes u. The Helmholtz projection solves this by introducing two Lagrange multipliers, lambda_1 and lambda_2, and finding the closest function u_star (in the Helmholtz-weighted norm) that satisfies both C1 and C2.

The forward pass works as follows. A Newton iteration solves for the two multipliers. At each Newton step, it builds a tridiagonal Helmholtz operator whose coefficients depend on lambda_2 and the dispersion parameter mu, then solves the tridiagonal system using the Thomas algorithm (a standard O(N) forward-backward sweep). It evaluates the constraint residuals F1 and F2, computes the 2x2 Jacobian by solving two additional tridiagonal systems for the sensitivities, and updates the multipliers. Convergence to machine precision typically takes fewer than ten iterations.

The backward pass uses the implicit function theorem. Rather than differentiating through the Newton loop (which would require storing all intermediate iterates), it solves a single adjoint tridiagonal system. This gives the gradient of the projected solution with respect to the network output in O(N) time and O(N) memory, independent of the number of Newton iterations.

The 2D polynomial projection for KP follows the same affine-rescaling logic applied to double integrals over the spatial domain.


## train.py

The Trainer class manages the full training procedure. At initialisation, it generates Latin hypercube collocation points for the PDE residual, sets up a conservation evaluation grid, computes the reference invariants from the initial condition, and saves the run configuration to disk.

The PDE residual functions sit at module level (outside the class). Each one uses torch.autograd.grad to compute the required partial derivatives by automatic differentiation, then assembles the residual expression for the corresponding PDE. The KdV residual needs u_t, u_x, and u_xxx. The RLW residual needs u_t, u_x, and u_xxt. The KP residual first computes the inner expression (the KdV-like part), differentiates it with respect to x, and adds sigma times u_yy.

Training proceeds in two phases. The Adam phase runs for a specified number of iterations with a fixed learning rate. The L-BFGS phase follows, using the strong Wolfe line search. Both phases optionally add a conservation loss: at each step, a set of random time values is drawn, the network is evaluated on the conservation grid at those times, the projection is applied, and the residual conservation error (after projection, only the soft I3 penalty if enabled) is added to the loss. Checkpoints are saved at regular intervals and whenever the PDE loss reaches a new minimum.

Every 500 steps, the trainer evaluates the three conservation integrals at the midpoint of the time domain and records the relative drift from the reference values. All metrics go to the history log and, if available, to Weights and Biases for live monitoring.

The evaluate method computes the L2 absolute error, L2 relative error, and L-infinity error against the exact solution at the final time, along with conservation drift at the final time. It saves a NumPy archive containing the predicted and exact solution arrays for later plotting.

The method `is_run_valid` checks whether the best recorded loss is finite. If the loss is NaN or Inf (indicating a diverged run), the method returns false. The notebook uses this to decide whether to push results to GitHub. Any run that finishes with a finite loss is kept, regardless of how large that loss is, because even poor results are useful for ablation comparison.


## How a notebook ties it together

A typical 1D notebook (take the RLW single soliton as the example) begins by installing the library from GitHub and loading the Kaggle secrets for W&B and GitHub access. The configuration cell sets all hyperparameters: the physical parameters (epsilon, mu, soliton amplitude c, domain boundaries, final time), the network architecture (number of hidden layers, width), the training schedule (number of Adam iterations, number of L-BFGS iterations, conservation penalty weight), and the experiment name. Every parameter is visible and editable in one place.

The model-building cell creates the MLP, wraps it in HardBC1D with the soliton initial condition, and moves it to the GPU. It then instantiates the Trainer with the full configuration dictionary, which triggers collocation point generation, reference invariant computation, and config saving.

The training cell calls `trainer.train_adam` followed by `trainer.train_lbfgs`, with the appropriate conservation penalty weight for the chosen variant. A vanilla run uses zero penalty and no projection. A hard-BC run uses HardBC1D but no projection. A CE-PINN run uses HardBC1D with polynomial or Helmholtz projection and an optional soft penalty on the third invariant.

The evaluation cell calls `trainer.evaluate` with the exact solution function, prints the errors and conservation drift, and closes the W&B run.

The final cell checks `trainer.is_run_valid()`. If the run succeeded and the notebook is running on Kaggle, it clones the repository, copies the results into the appropriate directory, commits, and pushes. If the run diverged or is running locally, it skips the push. This means a crashed or diverged notebook never contaminates the results archive.

The 2D notebooks (KP) follow the same pattern but build their own training loop inline instead of using the Trainer class, because the KP equation takes three spatial inputs and the Trainer is written for 1D problems. The structure is identical: LHS collocation, Adam loop with manual gradient descent, evaluation against the exact solution, and gated push.
