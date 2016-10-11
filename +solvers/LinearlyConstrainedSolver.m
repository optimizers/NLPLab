classdef LinearlyConstrainedSolver < solvers.NLPSolver

  methods

    % Constructor
    function self = LinearlyConstrainedSolver(nlp, varargin)
      if nlp.m > sum(nlp.linear)
        error('Problem must have linear constraints only');
      end
      self = self@solvers.NLPSolver(nlp, varargin{:});
    end

  end  % methods

end
