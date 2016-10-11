classdef UnconstrainedSolver < solvers.NLPSolver

  methods

    % Constructor
    function self = UnconstrainedSolver(nlp, varargin)
      if nlp.m > 0 | sum(nlp.jFre) ~= nlp.n
        error('Problem must be unconstrained');
      end
      self = self@solvers.NLPSolver(nlp, varargin{:});
    end

  end  % methods

end
