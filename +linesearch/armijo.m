function [xNew, fNew, failed, t] = armijo(solver, x, f, g, d)
%% Armijo - Armijo line search
% Inputs:
%   - solver: NlpSolver object. Must possess suffDec & maxIterLS
%   properties.
%   - x: current point
%   - f: objective function value at x
%   - g: gradient at x
%   - d: current descent direction
% Ouputs:
%   - xNew: new point upon Armijo's termination
%   - fNew: objective function value at xNew
%   - failed: failure flag (true if failed, false if convergence)
%   - t: step length (might be needed by some solvers)

iterLS = 1;
t = 1;
failed = false;
while true
    % Recompute trial step on free variables
    xNew = x + t * d;
    % Update objective function value
    fNew = solver.nlp.obj(xNew);
    % Checking exit conditions
    if fNew <= f + (solver.suffDec * t * g' * d)
        % Armijo condition satisfied
        return;
    elseif iterLS >= solver.maxIterLS
        % Maximal number of iterations reached, abort
        failed = true;
        return;
    end
    % Decrease step size
    t = t / 2;
    iterLS = iterLS + 1;
end
end