classdef NlpSolver < handle
    %% NlpSolver
    % General NLP model solver. Solve function must be defined in subclass.
    
    
    properties (SetAccess = private, Hidden = false)
        nlp         % original nlp object
        name        % solver name
        aFeasTol   % absolute stopping tolerance on feasibility
        aOptTol    % absolute stopping tolerance on optimality
        aCompTol   % absolute stopping tolerance on complementarity
        maxIter    % maximum number of iterations
        gradCheck  % activate gradient checker
    end % gettable private properties
    
    properties (Access = public)
        x;          % current point upon termination
        fx;         % objective function value at x
        rOptTol    % relative stopping tolerance on optimality
        rFeasTol   % relative stopping tolerance on feasibility
        rCompTol   % relative stopping tolerance on complementarity
        solved      % bool flag that indicates if problem has been solved
        nObjFunc; % # of obj func calls
        nGrad; % # of gradient calls
        nHess; % # of hessian calls
        solveTime  % cpu time to complete the solve
        status      % status after solve
        iter        % current iteration count
        logger      % logger object
    end
    
    properties (Hidden = true, Constant)
        % TO DO: figure those out
%         EXIT_OPTIMAL    = 0;
%         EXIT_ITERATIONS = 1;
%         EXIT_UNKNOWN    = 999;
%         EXIT_MSG = {
%             'Optimal solution found'
%             'Too many iterations'
%             'Unknown termination condition'
%             };
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
            p.addParameter('name',       'generic');
            p.addParameter('aFeasTol',  1.0e-8);
            p.addParameter('rFeasTol',  1.0e-6);
            p.addParameter('aOptTol',   1.0e-8);
            p.addParameter('rOptTol',   1.0e-6);
            p.addParameter('aCompTol',  1.0e-8);
            p.addParameter('rCompTol',  1.0e-6);
            p.addParameter('maxIter',   max(50, min(2*nlp.n, 200)));
            p.addParameter('gradCheck', false);
            p.addParameter('logger',     0);
            
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