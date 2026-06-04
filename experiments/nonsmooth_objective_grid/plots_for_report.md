### euclidian prox funtion on benchamark objective functions

## iter vs objective gap plots (trajectories)

SDA simple, gamma mult 1, unrestricted, objective_gap_x_hat , all D vals, all objectives ; diff D vals on one plot, separate plots for each objective

%default params plot we can see thet if minimum is not inside prox the algorihhm may NOT WORK (explanation), in this example algorith converage based on final gap < eps even though the real objective gap is high

SDA simple, gamma mult 1, restricted, objective_gap_x_hat , all D vals, all objectives ; diff D vals on one plot, separate plots for each objective

%for restricted case we change the set over which we minimize since we always are inside F_D the assumptions of alg is satisfied but we find the solution in F_D for problem defined on F_D which is less general than finding solution in F_D for problem defined in Q>F_D

SDA simple, gamma mult = 0.1, D=8. unrestricted vs restricted , both objective_gap_x and objective_gap_x_hat on one plot

%even if minimum is inside prox < D the algorithm behave differrently since for unrestricted version x can go outside prox to come back there later which is not possible for restricted so from now on we would focus only on unrestricted cases which is the intended way of using sda - this requires addition knowledge that the minimum is inside prox < D (D is parameter of algorithm)

SDA simple, all gamma mults, unrestricted, objective_gap_x_hat , D=8, all objectives ; diff gamma vals on one plot, separate plots for each objective

%here we set D=8 for all objectives to compare different gamma values (gamma_mul is a multiplier of nestrov propposed gamma value, we can see that proposed one. (gamma mult = 1) is not always the best in terms of convergance so it is another parameter to take care of)

SDA simple, gammamult [0.5, 2], unrestricted, objective_gap_x_hat , all D, all objectives ; diff gamma vals and D on one plot, separate plots for each objective

%one specific gamma mult can be good for specific D for for other D it can be bad so more important is rather combination of gamma + D (what odes gamma and D affect? and how they can behave together depending on how far from minimum we start and how these params can be chosen)

% we also test weighted sda and standard subgradient (in weighted rho= 1/ gamma, in subgradinet ...)

SDA simple , weighted, subgradient, gammamult = 1 alpha=1, unrestricted, objective_gap_x_hat , D=[2,8], all objectives ; diff D vals on one plot and diff methods, separate plots for each objective

% grid of all subgrad params, sda simple and weighter for last plot

runtime vs final objective gap -same as is
