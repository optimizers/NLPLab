classdef SolverStatus < handle

  properties
    iter          % Number of iterations
    iter_history  % Iterations history
    x             % Final iterate
    f             % Final objective value
    c             % Final vector of constraints
    kkt           % Final optimality residual
    msg           % Message
  end

end
