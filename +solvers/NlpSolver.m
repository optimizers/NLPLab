classdef NlpSolver < handle
    %% NlpSolver
    % General NLP model solver. Solve function must be defined in subclass.
    
    
    properties (SetAccess = private, Hidden = false)
        nlp;        % Original nlp object
        name;       % Solver name
        aFeasTol;   % Absolute stopping tolerance on feasibility
        aOptTol;    % Absolute stopping tolerance on optimality
        aCompTol;   % Absolute stopping tolerance on complementarity
        maxIter;    % Maximum number of iterations
        gradCheck;  % Activate gradient checker
        verbose;    % 0, 1 or 2
        maxEval;    % Maximum number of calls to objective function
    end % gettable private properties
    
    properties (Access = public)
        x;          % current point upon termination
        fx;         % objective function value at x
        pgNorm;     % Norm of projected gradient at x
        iStop;      % Exit flag
        rOptTol;    % relative stopping tolerance on optimality
        rFeasTol;   % relative stopping tolerance on feasibility
        rCompTol;   % relative stopping tolerance on complementarity
        solved;      % bool flag that indicates if problem has been solved
        nObjFunc; % # of obj func calls
        nGrad; % # of gradient calls
        nHess; % # of hessian calls
        solveTime;  % cpu time to complete the solve
        status;      % status after solve
        iter;        % current iteration count
        logger;      % logger object
    end
    
    
    methods (Access = public)
        
        function self = NlpSolver(nlp, varargin)
            %% Constructor
            
            % Verifying that nlp is a subclass of nlpmodel
            if ~isa(nlp, 'model.NlpModel')
                error('nlp should be a NlpModel');
            end
            
            self.nlp = nlp;
            
            % Define optional parameters and default values
            p = inputParser;
            p.PartialMatching = false;
            p.KeepUnmatched = false;
            p.addParameter('name',      'generic');
            p.addParameter('aFeasTol',  1.0e-12);
            p.addParameter('rFeasTol',  1.0e-6);
            p.addParameter('aOptTol',   1.0e-8);
            p.addParameter('rOptTol',   1.0e-6);
            p.addParameter('aCompTol',  1.0e-8);
            p.addParameter('rCompTol',  1.0e-6);
            p.addParameter('maxIter',   max(50, min(2*nlp.n, 200)));
            p.addParameter('maxEval',   nlp.n);
            p.addParameter('verbose',   1);
            p.addParameter('gradCheck', false);
            p.addParameter('logger',    0);
            
            % Parse optional parameters and store values
            p.parse(varargin{:});
            self.name = p.Results.name;
            self.aFeasTol = p.Results.aFeasTol;
            self.rFeasTol = p.Results.rFeasTol;
            self.aOptTol = p.Results.aOptTol;
            self.rOptTol = p.Results.rOptTol;
            self.aCompTol = p.Results.aCompTol;
            self.rCompTol = p.Results.rCompTol;
            self.maxIter = p.Results.maxIter;
            self.maxEval = p.Results.maxEval;
            self.verbose = p.Results.verbose;
            self.gradCheck = p.Results.gradCheck;
            self.logger = p.Results.logger;
            
            if self.logger == 0  % Disable logging
                import logging.logging
                logger_name = strcat(self.name, '-', nlp.name, '.log');
                opts.path = strcat(pwd, '/', logger_name);
                self.logger = logging.getLogger(logger_name, opts);
                
                self.logger.setCommandWindowLevel(logging.logging.OFF);
                self.logger.setLogLevel(logging.logging.OFF);
            end
            
            self.iStop = 0;
            self.iter = 0;
            self.status = 'unknown';
            self.solveTime = Inf;
        end
        
    end % public methods
    
    
    % List of abstract methods that must be defined in a subclass
    methods (Abstract)
        self = solve(self, varargin)
    end
    
end % class