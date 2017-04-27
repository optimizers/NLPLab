classdef IpoptSolver < solvers.NlpSolver
    %% IpoptSolver
    % Wrapper class that calls the IPOPT solver. Of course, IPOPT is
    % required in order to use this.
    % Get IPOPT: https://projects.coin-or.org/Ipopt
    % ***
    % Make sure that ipopt.mexa64 is on MATLAB's path!
    % Run or include on startup:
    % addpath('path/to/ipopt.mexa64');
    % ***
    
    properties (SetAccess = private, Hidden = false)
        funcs;
        options;
        info;
    end
    
    
    methods (Access = public)
        
        function self = IpoptSolver(nlp, varargin)
            %% Constructor
            
            % Checking if ipopt.mexa64 is on the path
            if ~exist('ipopt.mexa64', 'file')
                error('ipopt.mexa64 not on MATLAB''s path!');
            end
            
            p = inputParser;
            p.KeepUnmatched = true;
            % Guesses to the initial Lagrange multipliers (warm start)
            p.addParameter('zl', []);
            p.addParameter('zu', []);
            p.addParameter('lambda', []);
            
            p.parse(varargin{:});
            
            self = self@solvers.NlpSolver(nlp, p.Unmatched);
            
            %% See ipopt.m for more information about the following
            % Converting the NlpModel into the funcs & options structs
            self.funcs = struct;
            self.funcs.objective = @(x) self.nlp.fobj(x);
            self.funcs.gradient = @(x) self.nlp.gobj(x);
            self.funcs.constraints = @(x) self.nlp.fcon(x);
            self.funcs.jacobian = @(x) self.nlp.gcon(x);
            self.funcs.jacobianstructure = @() self.nlp.Jpattern;
            self.funcs.hessian = @(x, sigma, lambda) tril(self.nlp.hlag(x, ...
                lambda));
            self.funcs.hessianstructure = @() tril(self.nlp.Hpattern);
            % Callback routine, once per iteration. Receives (x, fx, info),
            % where info is a struct.
            self.funcs.iterfunc = @(iter, fx, info) ...
                self.iterFunc(iter, fx, info);
            self.options.lb = self.nlp.bL;
            self.options.ub = self.nlp.bU;
            self.options.cl = self.nlp.cL;
            self.options.cu = self.nlp.cU;
            
            % Initial guesses of the Lagrange multiplier (warm start)
            if ~isempty(p.Results.zl), self.options.zl = p.Results.zl; end
            if ~isempty(p.Results.zu), self.options.zu = p.Results.zu; end
            if ~isempty(p.Results.lambda)
                self.options.lambda = p.Results.lambda;
            end
            
            % Additional IPOPT parameters
            % See https://www.coin-or.org/Ipopt/documentation/node40.html
            % There are many more parameters, we only set the following
            self.options.ipopt = struct;
            % IPOPT uses 12 levels of verbose
            % Note that the output of the solver is explained at
            % https://www.coin-or.org/Ipopt/documentation/node36.html
            if self.verbose == 0
                self.options.ipopt.print_level = 0;
            elseif self.verbose == 1
                self.options.ipopt.print_level = 3;
            else
               self.options.ipopt.print_level = 5; 
            end
            self.options.ipopt.tol = self.aOptTol;
            self.options.ipopt.max_iter = self.maxIter;
            self.options.ipopt.max_cpu_time = self.maxRT;
            self.options.ipopt.dual_inf_tol = self.aCompTol;
            self.options.ipopt.constr_viol_tol = self.aOptTol;
            self.options.ipopt.compl_inf_tol = self.aCompTol;
            self.options.ipopt.obj_scaling_factor = self.nlp.obj_scale;
            self.options.ipopt.nlp_scaling_method = 'none';
            self.options.ipopt.acceptable_iter = 0;
        end
        
        function self = solve(self)
            %% Solve - call the IPOPT solver
            [x, info] = ipopt(self.nlp.x0, self.funcs, self.options);
            
            % Gathering information ...
            self.info = info;
            self.x = x;
            self.iter = info.iter;
            self.solveTime = info.cpu;
            self.nObjFunc = info.eval.objective;
            self.nGrad = info.eval.gradient;
            self.nHess = info.eval.hessian;
            
            
            if info.status == 0 || info.status == 1
                self.iStop = 1;
            elseif info.status == 3 
                self.iStop = 12;
            elseif info.status == -1
                self.iStop = 4;
            else
               self.iStop = 14; 
            end
            
            self.isSolved();
        end
        
    end
    
    methods (Access = private)
        function flag = iterFunc(self, iter, fx, info)
           %% IterFunc - callback function for IPOPT
           % Called once per iteration, used to store and update the
           % information output by IPOPT.
           % Inputs:
           %    - x: current iterate
           %    - fx: current value of the objective function
           %    - info: struct containing the following fields: inf_pr, 
           %    inf_du, mu, d_norm, regularization_size, alpha_du, 
           %    alpha_pr and ls_trials
           % Ouput:
           %    - flag: true, unless you want the algorithm to stop
           %    prematurely
           
           self.iter = iter;
           self.x = info.x;
           self.fx = fx;
           self.pgNorm = max(info.inf_pr, info.inf_du);
           flag = true; % always continue
        end
    end
    
end