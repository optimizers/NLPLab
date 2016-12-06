classdef LbfgsbSolver < solvers.NlpSolver
    %% LbfgsbSolver - Calls the L-BFGS-B MEX interface solver
    % L-BFGS-B repository must be on MATLAB's path.
    % L-BFGS-B C++ MEX interface available at
    % https://bitbucket.org/polylion/optimization
    
    
    methods (Access = public)
        
        function self = LbfgsbSolver(nlp, varargin)
            self = self@solvers.NlpSolver(nlp, varargin{:});
        end
        
        function self = solve(self)
            %% Solve problem represented by nlp model using L-BFGS-B
            
            % Handle that returns nlp's objective value & gradient
            fgFunc = @(x, p) self.objSupp(x, p);
            
            options.stop_crit = self.options.ftol * ...
                norm(self.gobj_local(self.x0), inf);
            
            % Calling L-BFGS-B
            [zProj, pout, fs, stop_reason, nbfg, rt, iter_hist] = ...
                lbfgsb(self.x0, fgFunc, [], self.bL, self.bU, ...
                options.ftol, self.options.stop_crit, ...
                options.nb_corr, self.options.max_iter, ...
                self.options.max_rt, self.options.max_fg, ...
                self.options.verbosity, self.options.post_processing, ...
                self.options.iter_guard, self.options.start_attempt);
            
            if ~strcmp(stop_reason(1:4), 'CONV')
                warning('Projection sub-problem didn''t converge: %s', ...
                    stop_reason);
            end
            
            self.pout = pout;
            self.fs = fs;
            self.stop_reason = stop_reason;
            self.nbfg = nbfg;
            self.time_total = rt;
            self.iter_hist = iter_hist;
            self.proj_grad_norm = iter_hist(end, 3);
            self.iter = iter_hist(end, 1);
            
            self.x = zProj;
            
            fprintf('\nEXIT L-BFGS-B: %s\n', self.stop_reason);
            fprintf('||Pg|| = %8.1e\n', self.proj_grad_norm);
        end
        
    end % public methods
    
    
    methods (Access = private)
        
        function [x, g, p] = objSupp(self, x, p)
            %% Calling ProjModel's obj function and returning p
            [x, g] = self.nlp.obj(x);    
        end
        
    end % private methods
    
end % class