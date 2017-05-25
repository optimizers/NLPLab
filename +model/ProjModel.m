classdef ProjModel < model.LeastSquaresModel
    %% ProjModel - Custom model representing the projection problem
    % ProjModel is a regular LeastSquaresModel with a projection operation
    % and a modified hobjprod function. This class represents the DUAL of
    % the projection sub-problem:
    %
    %   {x | C*x >= 0}
    %
    % which comes to a bounded least squares problem :
    %
    %   min     1/2 || C'*z + \bar{x} ||^2
    %     z     z >= 0
    %
    % where z is a lagrange multiplier and C is the scaling matrix used in
    % the reconstruction problem.
    
    
    %% Properties
    properties (SetAccess = private, Hidden = false)
        CCt
    end
    
    
    %% Public methods
    methods (Access = public)
        
        function self = ProjModel(CCt, C, objSize)
            %% Constructor
            %   min     1/2 || C'*z + \bar{x} ||^2
            %     z     z >= 0
            
            % This problem only has a lower bound.
            empt = sparse(objSize, objSize); % the Jacobian
            bU = inf(objSize, 1);
            bL = zeros(objSize, 1);
            
            % Call LeastSquaresModel : (A, b, C, cL, cU, bL, bU, x0, name)
            % A is our C
            % b is our -xbar, which will be set upon call to project
            % C is the Jacobian of the linear constraints (empt)
            % cL, cU are unbounded (-inf and inf)
            % bL is zero
            % bU is inf
            % x0 is zeros
            self = self@model.LeastSquaresModel(C, bL, empt, -bU, bU, ...
                bL, bU, bL);
            
            % We keep a handle on the efficient computation of C'*C
            self.CCt = CCt;
        end
        
        function [z, solved] = project(self, z)
            %% Projects z into { z | z >= 0 }
            z = max(z, self.bL);
            solved = true;
        end
        
        %% The following functions are redefined from the parent class
        function hess = hobj_local(self, z)
            %% Computes the hessian of the proj. obj. func.
            hess = self.CCt;
        end
        
        function Hv = hobjprod_local(self, ~, ~, v)
            %% Computes the hessian of proj. prob. obj. func. times vector
            Hv = self.CCt * v;
        end
    end
    
end