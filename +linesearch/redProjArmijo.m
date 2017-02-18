function [xNew, fNew, failed, t] = redProjArmijo(solver, xNew, x, f, ...
    g, d, working)
%% RedProjArmijo - Projected Armijo line search on the restricted vars
% Perform a projected Armijo line search on the reduced
% variables according to 'working'. This function assumes that
% x and g are already reduced. However, xNew
% must be full-sized since calls will be made to the objective
% function.
% !!!
% Only works with bounded problems!
% !!!
% Inputs:
%   - solver: NlpSolver object. Must possess suffDec & maxIterLS
%   properties.
%   - x: current point
%   - f: objective function value at x
%   - g: gradient at x
%   - d: current descent direction
%   - working: logical array of working (free) variables at x
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
    xNew(working) = solver.projectSel(x + t * d, working);
    % Update objective function value
    fNew = solver.nlp.obj(xNew);
    % Checking exit conditions
    if (f - fNew) >= (solver.suffDec * t * g' * d)
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