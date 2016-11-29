classdef MinConfProjModel < model.ProjModel
    %% MinConfProjModel - Custom model representing the projection problem
    %   The Solve function calls minConf_TMP on the functions specifically
    %   defined in this class to represent the dual of the projection
    %   problem.
    %
    %   This class was developped for the following problem
    %   min   1/2 || C'z + \bar{x} ||^2
    %   s.c.  z >= 0
    %   where z is the lagrange multiplier. The previous problem is
    %   the dual of the projection problem {x | Cx >= 0}.
    %
    %   This model is only compatible with the structure of the tomographic
    %   reconstruction algorithm made by Poly-LION. The preconditionner
    %   should be a Precond object.
    %   ---
    
    properties (SetAccess = private, Hidden = false)
        % minConf parameters
        options;
        % Internal parameters
        xbar;
        xProj;
    end
    
    methods (Access = public)
        function self = MinConfProjModel(prec, geos, options)
            %% Constructor
            % Inputs:
            %   - varargin: struct object or multiple arguments
            %   representing minConf_TMP's parameters.
            
            % Calling ProjModel's constructor
            self = self@model.ProjModel(prec, geos);
            % Update the default values
            self.options = options;
        end
        
        function [xProj, solver] = Solve(self, x)
            %% Projection function for the convex problem
            % Second function that is passed to minConf_PQN, calculates the
            % projection of a point x on the constraint set Cx >= 0.
            %
            % To obtain this projection, we need to solve the problem
            % {x | C*x >= 0}. However, we think it might be easier to solve
            % the dual of this problem, which is described in projProbFunc.
            
            % Variable that we desire to project on the constraint set
            % Here x is \bar{x}
            self.xbar = x;
            
            % Solving the projection problem
            solver = MinConf_TMP(self, self.options);
            if ~solver.solved
                warning('Projection failed: exit on non-optimal value');
            end
            zProj = solver.x;
            xProj = self.dualToPrimal(zProj);
            self.xProj = xProj;
        end
    end
end
