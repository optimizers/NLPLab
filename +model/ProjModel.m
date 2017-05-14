classdef ProjModel < model.LeastSquaresModel
    %% ProjModel - Custom model representing the projection problem
    %   This class represents the DUAL of the projection sub-problem:
    %
    %   {x | C*x >= 0}
    %
    %   This sub-problem is encountered when projections on the constraint 
    %   set are required to solve the tomographic reconstruction problem 
    %   defined by RecModel. Solving the dual comes to a bounded problem :
    %
    %   min     1/2 || C'*z + \bar{x} ||^2
    %     z     z >= 0
    %
    %   where z is a lagrange multiplier, C is the scaling matrix used in 
    %   the reconstruction problem, and is much easier to solve than the
    %   primal (original) projection problem :
    %
    %   min     1/2 || x - \bar{x} ||^2
    %     x     C*x >= 0   
    %
    %   This model is only compatible with the structure of the tomographic
    %   reconstruction algorithm made by Poly-LION. The scaling matrix C
    %   should be a Precond object.
    
    
    %% Properties
    properties (SetAccess = private, Hidden = false)
        objSize; % Real object size according to GeoS object
        prec;
        AAt;
        normJac;
        xbar;
        solved = true;
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
            b = sparse(objSiz, 1, 0);
            
            % A will depend on current object's function precMult, passing
            % a temporary value to the constructor
            empt = sparse(objSiz, objSiz, 0);
            
            self = self@model.LeastSquaresModel(empt, b, empt, -bU, bU, ...
                bL, bU, z0);
            
            % Initializing input parameters
            self.objSize = objSiz;
            self.prec = prec;
            
            % Updating A once the object exists
            self.A = opFunction(self.m, self.n, ...
                @(z, mode) self.precMult(z, mode));
            self.AAt = opFunction(self.m, self.m, ...
                @(z, mode) self.hobjprod([], [], z));
            % Getting the norm of the preconditionner, helps to evaluate
            % "relative" decreases/zeros.
            self.normJac = self.prec.norm();
        end
        
        function setPointToProject(self, xbar)
            %% Set xbar as b in the obj. func. 1/2 * || A*x - b ||^2
            self.b = -xbar;
            self.xbar = xbar;
        end
        
        function xProj = dualToPrimal(self, zProj)
            %% Retrieving the original variable: x = \bar{x} + C'*z
            xProj = self.xbar + real(self.prec.Adjoint(zProj));
        end
        
        function z = project(self, z)
            %% Projects { z | zProj >= 0 }
            z = max(z, self.bL);
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
        
        function z = hessPrecBCCB(self, ind, v)
            %% HessPrecProd
            % This function returns the product of
            %   diag(B * C * C' * B')⁻¹ * v
            % as a preconditionner to the hessian of the projection
            % sub-problem C * C'.
            % Inputs:
            %   - ind: logical array representing the restriction matrix B
            %   - v: arbitrary vector that is already of reduced size
            % Output:
            %   - z: product of preconditionner times v
            
            % Converting logical to positions
            ind = find(ind);
            
            % r & c hold the row and column to which each ind correspond in
            % block form (block size, # blocks)
            [r, c] = ind2sub([self.prec.BlkSiz, self.prec.Nblks], ind);
            
            % Since C * C' is block-circulant, in the worst case, we need
            % to form one full block to obtain the complete diagonal of
            % C * C'. Therefore, we can remove duplicates.
            [r, ~, iR] = unique(r);
            nR = length(r);
            
            % Initialize the diagonal
            dC = zeros(nR, 1);
            for ii = 1 : nR
                % For each r, create a row of B
                p = zeros(1, self.prec.Nblks);
                p(c(ii)) = 1;
                
                % (B * F' * D² * F * B)_ith row of block form
                temp = ifft( ...
                    self.prec.Mdiag(r(ii) : self.prec.BlkSiz : end).^2 ...
                    .* fft(p, [], 2), [], 2);
                
                % Only extract the diagonal value of the block products
                dC(ii) = temp(c(ii));
            end
            
            % Inverting single block before expanding
            dC = 1./dC;
            % Expanding on duplicate indices
            z = dC(iR) .* v;
        end
        
        function z = hessPrecD(self, ind, v)
            %% HessPrecD
            z = (1./(self.prec.Mdiag(ind).^2)) .* v;
        end
        
    end
    
    
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