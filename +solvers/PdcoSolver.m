classdef PdcoSolver < solvers.LinearlyConstrainedSolver

  properties (SetAccess=private)
  end

  methods

    function self = PdcoSolver(nlp, varargin)
      self = self@solvers.LinearlyConstrainedSolver(nlp, varargin{:});
    end

  end

end
