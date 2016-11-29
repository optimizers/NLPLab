classdef ProjModel < model.nlpmodel
    %% ProjModel - Custom model representing the projection problem
    %   This class represents the DUAL of the projection sub-problem:
    %
    %   {x | C*x >= 0}
    %
    %   This sub-problem is encountered when either minConf or Cflash are
    %   used to solve the tomographic reconstruction problem defined by
    %   RecModel. Generally speaking, it is much easier to solve the dual
    %   of the aforementioned problem. Therefore, we will solve the
    %   following problem instead (dual):
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
        % Storing the objects representing the problem
        prec; % The preconditionner used, identity = none
        % Internal parameters
        objSize; % Real object size according to GeoS object
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    methods (Abstract)
        [xSol, solver] = Solve(self, x);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    methods (Access = public)
        
        function self = ProjModel(prec, geos)
            %% Constructor
            % Inputs:
            %   - prec: Precond object from the Poly-LION repository
            %   - geos: Geometrie_Series object from the Poly-LION repo
            
            % Checking the arguments
            if ~isa(prec, 'Precond')
                error('Object prec should be a Precond object');
            elseif ~isa(geos, 'GeometrieSeries')
                error('Object geos should be a GeometrieSeries object');
            end
            
            % Getting object size depending on coordinates type
            if isa(geos, 'GeoS_Cart')
                objSiz = geos.Rows * geos.Columns * geos.Slices;
            else % Has to be GeoS_Pol
                objSiz = geos.Rhos * geos.Thetas * geos.Slices;
            end
            
            % Initial lagrange multiplier
            z0 = zeros(objSiz, 1); % Watch out, z0 becomes x0 in nlpmodel
            % Upper and lower bounds of the projection problem
            bU = inf(objSiz, 1);
            bL = zeros(objSiz, 1);
            
            % Calling the nlpmodel superclass (required for bcflash)
            self = self@model.nlpmodel('', z0, [], [], bL, bU);
            
            % Initializing input parameters
            self.prec = prec;
            self.objSize = objSiz;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %               --- Projection problem functions ---
        % Override NLP model's default functions
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [fObj, grad, hess] = obj(self, z)
            %% Function returning the obj. func, grad. and hess.
            % min   1/2 || C'z + \bar{x} ||^2
            % <=>
            % min   1/2 z'*C*C'*z + 1/2 \bar{x}'*\bar{x} + z'*C*\bar{x}
            %  z
            % s.c.  z >= 0
            % where z is the lagrange multiplier. The previous problem is
            % the dual of the projection problem {x | Cx >= 0}.
            %
            % The primal problem is
            % min   1/2 || x - \bar{x} ||^2
            %  x
            % s.c.  C*x >= 0
            
            if nargout == 1
                fObj = self.fobj(z);
            elseif nargout == 2
                [fObj, grad] = self.fgobj(z);
            else % nargout == 3
                [fObj, grad] = self.fgobj(z);
                % Build a Spot op such that its product is
                % hessian(z) * v
                hess = self.hobj(z);
            end
        end
        
        function f = fobj_local(self, z)
            %% Computes the obj. func. of the projection problem
            f = real(0.5*(z'*self.prec.Direct(self.prec.Adjoint(z) + ...
                2*self.xbar) + self.xbar'*self.xbar));
        end
        
        function g = gobj_local(self, z)
            %% Computes the gradient of the projeciton problems
            g = real(self.prec.Direct(self.prec.Adjoint(z) + self.xbar));
        end
        
        function [f, g] = fgobj_local(self, z)
            %% "Efficient" computation of the gradient and obj. func.
            % The purpose of this method is to reduce the amount of
            % computations required by reusing part of the computations
            % already made.
            res = self.prec.Adjoint(z) + self.xbar;
            f = real(0.5 * (res' * res));
            g = real(self.prec.Direct(res));
        end
        
        function hess = hobj_local(self, z)
            %% Computes the hessian of the proj. obj. func.
            % Hessian is symetric, so there is no need to carry the mode
            % argument
            % This Spot operator is rebuilt everytime a call to the hessian
            % is made even though it doesn't depend on z. This format is
            % kept to maintain consistency throughout the code
            hessWrap = @(v, mode) self.hobjprod(z, [], v);
            hess = opFunction(self.objSize, self.objSize, hessWrap);
        end
        
        function H = hobjprod_local(self, ~, ~, v)
            %% Computes the hessian of proj. prob. obj. func. times vector
            % This hessian will never depend on z (second argument),
            % however the original format is maintained to stay consistant
            % throughout the code
            H = real(self.prec.AdjointDirect(v));
        end
        
        function H = hlag_local(self, x, ~)
            %% Computes the hessian of the lagrangian
            % Considering all constraints are linear, the hessian of the
            % lagrangian is the hessian of the objective function
            H = self.hobj_local(x);
        end
        
        function w = hlagprod_local(self, ~, ~, v)
            %% Computes the hessian of the lagrangian times vector
            % bcflash doesn't require a Spot operator. We directly call
            % hobj_prod. The hessian of the lagrangian of the problem is
            % the hessian of the objective function, CC'.
            w = self.hobjprod_local([], [], v);
        end
        
        function xProj = dualToPrimal(self, zProj)
            %% Retrieving the original variable: x = \bar{x} + C'*z
            xProj = self.xbar + real(self.prec.Adjoint(zProj));
        end
    end
end