classdef ProjModel < model.LeastSquaresModel
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
    
    
    %% Properties
    properties (SetAccess = private, Hidden = false)
        objSize; % Real object size according to GeoS object
        prec;
    end
    
    
    %% Public methods
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
            z0 = zeros(objSiz, 1); % Watch out, z0 becomes x0 in NlpModel
            % Upper and lower bounds of the projection problem
            bU = inf(objSiz, 1);
            bL = zeros(objSiz, 1);
            
            % !!
            % b (the vector that we wish to project) is undefined for now,
            % it must be set prior to the call to solve() using
            % setPointToProject.
            % !!
            b = [];
            
            % A will depend on current object's function precMult, passing
            % a temporary value to the constructor
            empt = sparse(objSiz, objSiz, 0);
            
            self = self@model.LeastSquaresModel(empt, b, empt, [], [], ...
                bL, bU, z0);
            
            % Initializing input parameters
            self.objSize = objSiz;
            self.prec = prec;
            
            % Updating A once the object exists
            self.A = opFunction(objSiz, objSiz, ...
                @(z, mode) self.precMult(z, mode));
        end
        
        function setPointToProject(self, xbar)
            %% Set xbar as b in the obj. func. 1/2 * || A*x - b ||^2
            self.b = xbar;
        end
        
        function xProj = dualToPrimal(self, zProj)
            %% Retrieving the original variable: x = \bar{x} + C'*z
            xProj = self.xbar + real(self.prec.Adjoint(zProj));
        end
        
        function zProj = project(self, z)
            %% Projects { z | zProj >= 0 }
            zProj = max(z, self.bL); % min(max(z, self.bL), self.bU)
        end
        
        %% The following functions are redefined from the parent class
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
        
    end
    
    
    %% Private methods
    methods (Access = private)
        
        function z = precMult(self, z, mode)
            %% Evaluates C * z
            if mode == 1
                z = self.prec.Direct(z);
            elseif mode == 2
                z = self.prec.Adjoint(z);
            end
            z = real(z);
            self.ncalls_hvp = self.ncalls_hvp + 1;
        end
        
    end
    
end