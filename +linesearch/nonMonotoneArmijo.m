function [xNew, fNew, failed, t] = nonMonotoneArmijo(solver, x, f, g, d)
%% NmArmijo - Non-monotone Armijo Line Search with SPG step length
% Inputs:
%   - solver: NlpSolver object. Must possess suffDec, maxIterLS, memory and
%   storedObjFunc properties.
%   - x: current point
%   - f: objective function value at x
%   - g: gradient at x
%   - d: current descent direction
% Ouputs:
%   - xNew: new point upon Armijo's termination
%   - fNew: objective function value at xNew
%   - failed: failure flag (true if failed, false if convergence)
%   - t: step length (might be needed by some solvers)

% Redefine f as the maximum
fMax = max(solver.storedObjFunc);

iterLS = 1;
t = 1;
delta = g' * d;
failed = false;
while true
    
    xNew = x + t * d;
    fNew = solver.nlp.obj(xNew);
    
    if fNew <= fMax + solver.suffDec * t * delta
        % Armijo condition met
        return
    elseif iterLS >= solver.maxIterLS
        % Maximal number of iterations reached, abort
        failed = true;
        return;
    end
    
    t = t / 2;
    iterLS = iterLS + 1;
end
end