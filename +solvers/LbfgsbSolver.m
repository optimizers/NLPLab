classdef LbfgsbSolver < solvers.NlpSolver
    %% LbfgsbSolver - Calls the L-BFGS-B MEX interface solver
    % L-BFGS-B repository must be on MATLAB's path.
    % L-BFGS-B C++ MEX interface available at
    % https://bitbucket.org/polylion/optimization
    
    
    properties (SetAccess = private, Hidden = false)
        % Norm of the projected gradient at x
        pgNorm;
        verbose; % 0, 1, 2
        corrections;
        postProcessing = [];
        iterGuard = [];
        startAttempt = [];
        maxEval; % Maximum number of objective function evaluations
        maxRT;
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
            p.addParameter('verbose', 2);
            p.addParameter('corrections', 7);
            p.addParameter('maxEval', 5e2);
            p.addParameter('maxRT', 1e4);
            
            p.parse(varargin{:});
            
            self = self@solvers.NlpSolver(nlp, p.Unmatched);
            
            self.verbose = p.Results.verbose;
            self.corrections = p.Results.corrections;
            self.maxRT = p.Results.maxRT;
            self.maxEval = p.Results.maxEval;
        end
        
        function self = solve(self)
            %% Solve problem represented by nlp model using L-BFGS-B
            
            % Handle that returns nlp's objective value & gradient
            fgFunc = @(x, p) self.objSupp(x, p);
            
            % Relative optimality tolerance
            self.rOptTol = self.aOptTol * ...
                norm(self.nlp.gobj_local(self.nlp.x0), inf);
            
            
            % L-BFGS-B only handles verbose true or false, therefore
            % removing one from verbose and pass it to L-BFGS-B. That way
            % the final print can be displayed without the solver's log.
            verb = max(self.verbose - 1, 0);
            
            % Calling L-BFGS-B
            [zProj, ~, fs, stopReason, nFg, rt, iterHist] = ...
                lbfgsb(self.nlp.x0, fgFunc, [], self.nlp.bL, ...
                self.nlp.bU, self.aFeasTol, self.rOptTol, ...
                self.corrections, self.maxIter, self.maxRT, ...
                self.maxEval, verb, self.postProcessing, ...
                self.iterGuard, self.startAttempt);
            
            if ~strcmp(stopReason(1:4), 'CONV')
                warning('Projection sub-problem didn''t converge: %s', ...
                    stopReason);
            end
            
            % Collecting information from L-BFGS-B's output
            self.fx = fs;
            self.nObjFunc = nFg;
            self.nGrad = nFg;
            self.nHess = 0;
            self.solveTime = rt;
            self.pgNorm = iterHist(end, 3);
            self.iter = iterHist(end, 1);
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