classdef LbfgsbSolver < solvers.NlpSolver
    %% LbfgsbSolver - Calls the L-BFGS-B MEX interface solver
    % L-BFGS-B repository must be on MATLAB's path.
    % L-BFGS-B C++ MEX interface available at
    % https://bitbucket.org/polylion/optimization
    
    
    properties (SetAccess = private, Hidden = false)
        corrections;
        postProcessing = [];
        iterGuard = [];
        startAttempt = [];
    end
    
    
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
            
            p.parse(varargin{:});
            
            self = self@solvers.NlpSolver(nlp, p.Unmatched);

            self.corrections = p.Results.corrections;
        end
        
        function self = solve(self)
            %% Solve problem represented by nlp model using L-BFGS-B
            
            self.nlp.resetCounters();
            
            % Handle that returns nlp's objective value & gradient
            fgFunc = @(x, p) self.objSupp(x, p);
            
            % Relative optimality tolerance
            self.gNorm0 = norm(self.nlp.gobj_local(self.nlp.x0), inf);
            rOptTol = self.rOptTol * self.gNorm0;
                
            
            
            % L-BFGS-B only handles verbose true or false, therefore
            % removing one from verbose and pass it to L-BFGS-B. That way
            % the final print can be displayed without the solver's log.
            verb = max(self.verbose - 1, 0);
            
            % Calling L-BFGS-B
            [zProj, ~, fs, stopReason, ~, rt, iterHist] = ...
                lbfgsb(self.nlp.x0, fgFunc, [], self.nlp.bL, ...
                self.nlp.bU, self.aFeasTol, rOptTol + self.aOptTol, ...
                self.corrections, self.maxIter, self.maxRT, ...
                self.maxEval, verb, self.postProcessing, ...
                self.iterGuard, self.startAttempt);
            
            self.iStop = self.EXIT_OPT_TOL;
            self.solved = true;
            if strcmp(stopReason(1:2), 'RT') || ...
                    ~strcmp(stopReason(1:4), 'CONV') && ...
                    ~strcmp(stopReason(1:4), 'NCON')
                self.iStop = self.EXIT_INNER_FAIL;
                self.solved = false;
            end
            
            % Collecting information from L-BFGS-B's output
            self.fx = fs;
            % Cumulative counts
            self.nObjFunc = self.nlp.ncalls_fobj + self.nlp.ncalls_fcon;
            self.nGrad = self.nlp.ncalls_gobj + self.nlp.ncalls_gcon;
            self.nHess = self.nlp.ncalls_hvp + self.nlp.ncalls_hes;
            
            self.solveTime = rt;
            self.pgNorm = iterHist(end, 3);
            self.iter = iterHist(end, 1) + 1;
            self.x = zProj;
            
            if self.verbose
                fprintf('\nEXIT L-BFGS-B: %s\n', stopReason);
                fprintf('||Pg|| = %8.1e\n', self.pgNorm);
            end
        end
        
    end % public methods
    
    
    methods (Access = private)
        
        function [f, g, p] = objSupp(self, x, p)
            %% Calling ProjModel's obj function and returning p
            [f, g] = self.nlp.obj(x);
        end
        
    end % private methods
    
end % class