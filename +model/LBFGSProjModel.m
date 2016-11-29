classdef LBFGSProjModel < model.ProjModel
    %% LBFGSProjModel - Custom model representing the projection problem
    %   The Solve method of this model uses the L-BFGS-B algorithm
    %   (lbfgsb.m) in order to solve the DUAL problem of the
    %   projection sub-problem:
    %
    %   min   1/2 || C'*z + \bar{x} ||^2
    %     z   z >= 0
    %
    %   where z is a lagrange multiplier, C is the preconditionner used
    %   in the reconstruction problem.
    %
    %   This model is only compatible with the structure of the tomographic
    %   reconstruction algorithm made by Poly-LION. The preconditionner
    %   should be a Precond object.
    %   ---
    
    properties (SetAccess = private, Hidden = false)
        % -- Problem variables --
        xbar; % Vector to project
        xProj; % Projected vector
        % -- Internal parameters of the L-BFGS-B algorithm --
        options;
    end
    
    methods (Access = public)
        function self = LBFGSProjModel(prec, geos, varargin)
            %% Constructor
            % Inputs:
            %   - varargin: struct object or multiple arguments
            %   representing L-BFGS-B's parameters.
            
            % Calling ProjModel's constructor
            self = self@model.ProjModel(prec, geos);
            
            % We parse the arguments here because it's not done in lbfgsb.m
            % Set default values for L-BFGS-B's parameters
            p = inputParser();
            p.addParameter('ftol', 1e-10);
            p.addParameter('nb_corr', 7);
            p.addParameter('max_iter', 1e4);
            p.addParameter('max_rt', 1e10);
            p.addParameter('max_fg', 1.5e3);
            p.addParameter('verbosity', 0);
            p.addParameter('post_processing', []);
            p.addParameter('iter_guard', []);
            p.addParameter('start_attempt', []);
            
            % Parse the user input values
            p.parse(varargin{:});
            
            % Update the default values
            self.options = p.Results;
        end
        
        function [xProj, solver] = Solve(self, x)
            %% Solves the dual of the projection problem using L-BFGS-B
            % Calls lbfgsb.m feeding a function that evaluates the
            % objective function and the gradient of x according the the
            % model defined in this class.
            % Input:
            %   - x: vector to project on C*x >= 0
            % Output:
            %   - xProj: the projection of x on C*x >= 0
            
            % This solver is not an object! Return a struct
            solver = struct;
            
            % Variable that we desire to project on the constraint set
            % Here x is \bar{x}
            self.xbar = x;
            
            % Function required by L-BFGS-B that returns the objective
            % function value, the gradient and the supplemental parameters
            % p.
            fgFunc = @(x, p) self.objSupp(x, p);
            
            %             self.options.stop_crit = self.options.ftol;
            
            %             % Can also be inf norm
            self.options.stop_crit = self.options.ftol * ...
                norm(self.gobj_local(self.x0), inf);
            
            % Calling L-BFGS-B
            [zProj, pout, fs, stop_reason, nbfg, rt, iter_hist] = ...
                lbfgsb(self.x0, fgFunc, [], self.bL, self.bU, ...
                self.options.ftol, self.options.stop_crit, ...
                self.options.nb_corr, self.options.max_iter, ...
                self.options.max_rt, self.options.max_fg, ...
                self.options.verbosity, self.options.post_processing, ...
                self.options.iter_guard, self.options.start_attempt);
            
            if ~strcmp(stop_reason(1:4), 'CONV')
                warning('Projection sub-problem didn''t converge: %s', ...
                    stop_reason);
            end
            
            solver.pout = pout;
            solver.fs = fs;
            solver.stop_reason = stop_reason;
            solver.nbfg = nbfg;
            solver.time_total = rt;
            solver.iter_hist = iter_hist;
            solver.proj_grad_norm = iter_hist(end, 3);
            solver.iter = iter_hist(end, 1);
            
            % Finding the primal variable from the dual variable
            xProj = self.dualToPrimal(zProj);
            self.xProj = xProj;
            
            solver.x = xProj;
            fprintf('\nEXIT L-BFGS-B: %s\n', solver.stop_reason);
            fprintf('||Pg|| = %8.1e\n', solver.proj_grad_norm);
        end
    end
    
    methods (Access = private)
        function [x, g, p] = objSupp(self, x, p)
            %% Calling ProjModel's obj function and returning p
            [x, g] = self.obj(x);
        end
    end
end
