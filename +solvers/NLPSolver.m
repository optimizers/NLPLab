classdef NLPSolver < handle

  properties
    nlp         % original nlp object
    name        % solver name
    feas_atol   % absolute stopping tolerance on feasibility
    feas_rtol   % relative stopping tolerance on feasibility
    opt_atol    % absolute stopping tolerance on optimality
    opt_rtol    % relative stopping tolerance on optimality
    comp_atol   % absolute stopping tolerance on complementarity
    comp_rtol   % relative stopping tolerance on complementarity
    max_iter    % maximum number of iterations
    grad_check  % activate gradient checker
    iter        % current iteration count
    logger      % logger object
    status      % status after solve
    solve_time  % cpu time to complete the solve
  end

  properties (Hidden=true, Constant)
    EXIT_OPTIMAL    = 0;
    EXIT_ITERATIONS = 1;
    EXIT_UNKNOWN    = 999;
    EXIT_MSG = {
                'Optimal solution found'
                'Too many iterations'
                'Unknown termination condition'
               };
  end

  methods

    % Constructor
    function self = NLPSolver(nlp, varargin)

      self.nlp = nlp;

      % Define optional parameters and default values
      p = inputParser;
      p.KeepUnmatched = false;
      p.addParameter('name',       'generic');
      p.addParameter('feas_atol',  1.0e-8);
      p.addParameter('feas_rtol',  1.0e-6);
      p.addParameter('opt_atol',   1.0e-8);
      p.addParameter('opt_rtol',   1.0e-6);
      p.addParameter('comp_atol',  1.0e-8);
      p.addParameter('comp_rtol',  1.0e-6);
      p.addParameter('max_iter',   max(50, min(2*nlp.n, 200)));
      p.addParameter('grad_check', false);
      p.addParameter('logger',     0);

      % Parse optional parameters and store values
      p.parse(varargin{:});
      self.name       = p.Results.name;
      self.feas_atol  = p.Results.feas_atol;
      self.feas_rtol  = p.Results.feas_rtol;
      self.opt_atol   = p.Results.opt_atol;
      self.opt_rtol   = p.Results.opt_rtol;
      self.comp_atol  = p.Results.comp_atol;
      self.comp_rtol  = p.Results.comp_rtol;
      self.max_iter   = p.Results.max_iter;
      self.grad_check = p.Results.grad_check;
      self.logger     = p.Results.logger;

      if self.logger == 0  % Disable logging
        import logging.logging
        logger_name = strcat(self.name, '-', nlp.name, '.log');
        opts.path = strcat(pwd, '/', logger_name);
        self.logger = logging.getLogger(logger_name, opts);
                                        
        self.logger.setCommandWindowLevel(logging.logging.OFF);
        self.logger.setLogLevel(logging.logging.OFF);
      end

      self.iter       = 0;
      self.status     = 'unknown';
      self.solve_time = Inf;
    end

    function self = solve(self, varargin)
      self.logger.error('solve', 'This class must be subclassed');
      error('This class must be subclassed');
    end

  end  % methods

end  % classdef

