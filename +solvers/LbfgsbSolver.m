classdef LbfgsbSolver < solvers.NlpSolver
    %% LbfgsbSolver - Calls the L-BFGS-B MEX interface solver
    % L-BFGS-B repository must be on MATLAB's path.
    % L-BFGS-B C++ MEX interface available at
    % https://bitbucket.org/polylion/optimization
    
    
    properties (SetAccess = protected, Hidden = false)
        corrections;
        postProcessing = [];
        iterGuard = [];
        startAttempt = [];
        fid;
        fMin;
        hist;
    end
    
    properties (Hidden = true, Constant)
        % Minimal accepted condition number
        MINRCOND = 100 * eps;
        
        % Log header and body formats.
        LOG_HEADER_FORMAT = '';
        LOG_BODY_FORMAT = [];
        LOG_HEADER = {};
    end % constant properties
    
    
    methods (Access = public)
        
        function self = LbfgsbSolver(nlp, varargin)
            %% Constructor
            
            % Checking if lbfgsb.m is on the path
            if ~exist('lbfgsb', 'file')
                error('L-BFGS-B not on MATLAB''s path!');
            end
            
            p = inputParser;
            p.KeepUnmatched = true;
            p.addParameter('corrections', 7);
            p.addParameter('fid', 1);
            p.addParameter('fMin', -1e32);
            
            p.parse(varargin{:});
            
            self = self@solvers.NlpSolver(nlp, p.Unmatched);

            self.corrections = p.Results.corrections;
            self.fid = p.Results.fid;
            self.fMin = p.Results.fMin;
        end
        
        function self = solve(self)
            %% Solve problem represented by nlp model using L-BFGS-B
            
            self.nlp.resetCounters();
            self.solveTime = tic();
            
            printObj = utils.PrintInfo('L-BFGS-B (C interface)');
            
            % Handle that returns nlp's objective value & gradient
            fgFunc = @(x, p) self.objSupp(x, p);
            
            if self.verbose >= 2
                extra = containers.Map({'corrections', 'fMin'}, ...
                    {self.corrections, self.fMin});
                printObj.header(self, extra);
                self.printf(self.LOG_HEADER_FORMAT, self.LOG_HEADER{:});
            end
            
            % Relative optimality tolerance
            [f, g] = self.nlp.obj(self.nlp.x0);
            self.gNorm0 = norm(g, inf);
            optTol = self.rOptTol * self.gNorm0 + self.aOptTol;
            feasTol = self.rFeasTol * abs(f) + self.aFeasTol;
            
            % L-BFGS-B only handles verbose true or false, therefore
            % removing one from verbose and pass it to L-BFGS-B. That way
            % the final print can be displayed without the solver's log.
            %verb = max(self.verbose - 1, 0);
%             switch self.verbose
%                 case 0
            if self.verbose >= 2
                printEvery = 1;
            else
                printEvery = 0;
            end
            
            if self.logger.commandWindowLevel == self.logger.OFF
                verb = -1;
            elseif self.logger.commandWindowLevel >= self.logger.DEBUG
                verb = 100;
            else
                verb = 101;
            end
            
            % Calling L-BFGS-B
            
            [zProj, fs, info] = lbfgsb( ...
                self.nlp.x0, ...
                fgFunc, ...
                [], ...
                self.nlp.bL, ...
                self.nlp.bU, ...
                struct( 'nb_corr',   self.corrections, ...
                        'ftol',      feasTol, ...
                        'pgtol',     optTol, ...
                        'max_iter',  self.maxIter, ...
                        'max_fg',    self.maxEval, ...
                        'verbosity', verb, ...
                        'printEvery', printEvery, ...
                        'post_processing', {@(x, p)x(1) ; @(x,p)x(2)} )...
                );
            self.hist = info.err;
            switch info.taskInteger
                case 2
                    self.iStop = self.EXIT_MAX_ITER;
                case 3
                    self.iStop = self.EXIT_INNER_FAIL;
                case 21
                    self.iStop = self.EXIT_OPT_TOL;
                case 22
                    self.iStop = self.EXIT_FEAS_TOL;
                case 31
                    self.iStop = self.EXIT_MAX_RT;
                case 32
                    self.iStop = self.EXIT_MAX_EVAL;
                case 33
                    self.iStop = self.EXIT_DIR_DERIV;
                case {101, 102}
                    self.iStop = self.EXIT_STEP_SIZE;
                otherwise
                    self.iStop = self.EXIT_UNKNOWN;
            end
            
            % Collecting information from L-BFGS-B's output
            self.fx = fs;
            % Cumulative counts
            self.nObjFunc = self.nlp.ncalls_fobj + self.nlp.ncalls_fcon;
            self.nGrad = self.nlp.ncalls_gobj + self.nlp.ncalls_gcon;
            self.nHess = self.nlp.ncalls_hvp + self.nlp.ncalls_hes;
            
            self.pgNorm = info.err(end, 3);
            self.iter = info.iterations;
            self.x = zProj;
            
%             if self.verbose
%                 fprintf('\nEXIT L-BFGS-B: %s\n', self.EXIT_MSG{self.iStop});
%                 fprintf('||Pg|| = %8.1e\n', self.pgNorm);
%             end

            self.solveTime = toc(self.solveTime);
            self.isSolved();
            
            printObj.footer(self);
        end
        
    end % public methods
    
    
    methods (Access = private)
        
        function [f, g, p] = objSupp(self, x, p)
            %% Calling ProjModel's obj function and returning p
            [f, g] = self.nlp.obj(x);
        end
        
    end % private methods
    
    
    methods (Access = public, Hidden = true)
        
        function printf(self, varargin)
            fprintf(self.fid, varargin{:});
        end
        
    end % hidden public methods
    
end % class