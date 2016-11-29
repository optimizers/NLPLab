classdef BCFlashProjModel < model.ProjModel
    %% BCFlashProjModel - Custom model representing the projection problem
    %   The Solve method of this model uses the bcflash algorithm (bound
    %   constrained TRON) in order to solve the DUAL of the
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
        % bcflash parameters
        options;
        % Internal parameters
        xbar; % Vector to project
        xProj; % Projected vector
    end
    
    methods (Access = public)
        function self = BCFlashProjModel(prec, geos, options)
            %% Constructor
            % Calling the ProjModel constructor
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
            
            % If Solve is used to solve the projected gradient sub-problem,
            % we must restrict the Spot operators in order to represent the
            % active constraints.
            
            % Variable that we desire to project on the constraint set
            % Here x is \bar{x}
            self.xbar = x;
            
            % Calling the bcflash solver on this model
            solver = bcflash(self, self.options);
            if solver.eFlag == 2 || solver.eFlag == 6
                warning('Projection failed: exit on non-optimal value');
            end
            % Retrieving the solution
            zProj = solver.x;
            xProj = self.dualToPrimal(zProj);
            self.xProj = xProj;
        end
    end
end
