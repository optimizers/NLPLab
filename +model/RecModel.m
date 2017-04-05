classdef RecModel < model.NlpModel
    %% RecModel - Model representing the the reconstruction problem
    %   This class was developped to represent the following problem
    %
    %   min     1/2 || PCx - y ||^2 + \lambda\phi(Cx)
    %     x     Cx >= 0
    %
    %   where P and C are matrices that don't have an explicit form, i.e.
    %   only their matrix-vector product is available.
    %
    %   This class is a subclass of NLP model and can therefore be passed
    %   to solvers requiring NLP models (Cflash (TRON), PDCOO, etc).
    %
    %   This model is only compatible with the structure of the tomographic
    %   reconstruction algorithm made by Poly-LION. The terms in the
    %   objective function should be provided by a Critere object, whereas
    %   the preconditionner should be a Precond object. This model also
    %   expects the sinogram to be a Sinogramme object and the geos a
    %   GeometrieSeries object.
    
    
    %% Properties
    properties (SetAccess = private, Hidden = false)
        % Storing the object representing the problem
        crit; % Contains terms representing the objective function
        prec; % The preconditionner used, identity = none
        sino; % The sinogram of the problem
        geos; % The geometry that is used
        Jac; % opSpot for the Jacobian (constant, linear constraint)
        JacJact; % opSpot for Jacobian times the transpose of the Jacobian
        objSize; % number of variables
    end
    
    
    %% Public methods
    methods (Access = public)
        
        function self = RecModel(crit, prec, sino, geos, mu0, name)
            %% Constructor
            % Inputs:
            %   - crit: Critere object from the Poly-LION repository
            %   - prec: Precond object from the Poly-Lion repository
            %   - sino: Sinogramme object from the Poly-Lion repository
            %   - geos: Geometrie_Series object from the Poly-Lion repo
            %   - mu0: (optional) initial vector for the reconstruction
            %   problem
            %   - name: (optional) name of the nlp model
            
            % Checking the arguments
            if ~isa(crit, 'Critere')
                error('Object crit should be a Critere object');
            elseif ~isa(prec, 'Precond')
                error('Object prec should be a Precond object');
            elseif ~isa(sino, 'Sinogramme')
                error('Object sino should be a Sinogramme object');
            elseif ~isa(geos, 'GeometrieSeries')
                error('Object geos should be a GeometrieSeries object');
            end
            
            % Might change this later -- this should be in Critere
            nAdeq = 0;
            for ind = 1 : crit.nElemts
                if isa(crit.J{ind}, 'Adequation')
                    nAdeq = nAdeq + 1;
                end
            end
            if nAdeq ~= 1
                error(['There should be one Adequation term in the', ...
                    'critobject']);
            end
            
            % Getting object size depending on coordinates type
            if isa(geos, 'GeoS_Cart')
                objSiz = geos.Rows * geos.Columns * geos.Slices;
            else % Has to be GeoS_Pol
                objSiz = geos.Rhos * geos.Thetas * geos.Slices;
            end
            
            % In case of missing/empty arg or size mismatch
            if isempty(mu0)
                mu0 = spalloc(objSiz, 1, objSiz);
            elseif length(mu0(:)) ~= objSiz
                error(['Initial value vector''s size doesn''t match', ...
                    'the object'])
            end
            
            % Converting to the preconditionned variable
            % mu = Cx <=> x = C^-1 mu
            x0 = prec.Inverse(full(mu0));
            
            % Bounds of the problem. cU & cL represent the bounds on the
            % constraints and bU & bL represent the bounds on the
            % variables. Equalities are assumed to be the cases where
            % cU_i = cL_i.
            % Cx >= 0 is the only constraint
            cU = inf(objSiz, 1);
            cL = zeros(objSiz, 1);
            bL = -inf(objSiz, 1);
            bU = inf(objSiz, 1);
            
            % Calling the NlpModel superclass (required for PDCOO & Cflash)
            self = self@model.NlpModel(name, x0, cL, cU, bL, bU);
            
            % Constraints are linear
            self.linear = true(self.m, 1);
            
            % Assigning properties
            self.crit = crit;
            self.prec = prec;
            self.sino = sino;
            self.geos = geos;
            
            self.objSize = objSiz;
            
            % Some functions require Spot operators because only the
            % matrix-product is available.
            
            % opSpot to Jacobian of constraints
            self.Jac = opFunction(self.m, self.n, ...
                @(z, mode) self.precMult(z, mode));
            self.JacJact = opFunction(self.m, self.m, ...
                @(z, mode) real(self.prec.AdjointDirect(z)));
        end
        
        % Override the default NlpModel methods
        function [f, g, H] = obj(self, x)
            %% Evaluates the obj. func, the gradient and the hessian
            % The number of output arguments is set by the user.
            % Inputs:
            %   - x: vector
            % Ouputs:
            %   - f: f(x)
            %   - g: \nabla f(x)
            %   - H: \nabla^2 f(x)
            
            if nargout == 1
                % Only the objective function is required
                f = self.fobj(x);
            elseif nargout == 2
                % The gradient and the objective function are required.
                % Calling fgobj_local, which is optimized because it
                % requires a smaller amount of computations.
                [f, g] = self.fgobj(x);
            else % nargout == 3
                % The hessian, the gradient and the obj. func are required.
                % Calling fgobj_local, which is optimized because it
                % requires a smaller amount of computations.
                [f, g] = self.fgobj(x);
                % Build a Spot op such that its product is
                % hessian(z) * v
                H = self.hobj(x);
            end
        end
        
        function [fObj, grad] = fgobj_local(self, x)
            %% "Efficient" computation of the gradient and obj. func.
            % The purpose of this method is to reduce the amount of
            % computations required by reusing part of the computations
            % already made.
            x = real(x);
            fObj = 0;
            grad = zeros(size(x));
            for i = 1 : self.crit.nElemts
                vals = self.crit.J{i}.objGrad(x, reshape( ...
                    self.sino.Scans{1}, [], 1), [], self.prec);
                fObj = fObj + vals.fObj;
                grad = grad + vals.grad;
            end
            fObj = real(fObj);
            grad = real(grad);
        end
        
        function fObj = fobj_local(self, x)
            %% Computes the value of the objective function
            x = real(x);
            fObj = 0;
            for i = 1 : self.crit.nElemts
                fObj = fObj + self.crit.J{i}.objFunc(x, reshape( ...
                    self.sino.Scans{1}, [], 1), [], self.prec);
            end
            fObj = real(fObj);
        end
        
        function grad = gobj_local(self, x)
            %% Computes the value of the gradient of the obj. func.
            x = real(x);
            grad = zeros(size(x));
            for i = 1 : self.crit.nElemts
                grad = grad + self.crit.J{i}.grad(x, reshape( ...
                    self.sino.Scans{1}, [], 1), [], self.prec);
            end
            grad = real(grad);
        end
        
        function hess = hobj_local(self, x)
            %% Computes the value of the hessian of the obj. func.
            % Initializing the Spot operators required for the hessians
            % Wrappers to pass extra argument, required since hess might
            % depend on x.
            hessWrap = @(v, mode) self.hobjprod(x, [], v);
            hess = opFunction(self.objSize, self.objSize, hessWrap);
        end
        
        function hess = hobjprod_local(self, x, ~, v)
            %% Computes the hessian of the objective function times x
            x = real(x);
            hess = zeros(size(x));
            for i = 1 : self.crit.nElemts
                hess = hess + self.crit.J{i}.prodHess(x, reshape( ...
                    self.sino.Scans{1}, [], 1), v, self.prec);
            end
            hess = real(hess);
        end
        
        function c = fcon_local(self, x)
            %% Computes the value of the constraints
            % In our case the constraint is Cx >= 0
            c = real(self.prec.Direct(real(x)));
        end
        
        function J = gcon_local(self, ~)
            %% Computes the gradient of the constraints
            % In our case the constraint is Cx >= 0 so the jacobian is C
            J = self.Jac;
        end
        
        function Hc = hcon_local(self, ~, ~)
            %% Computes the hessian of the constraints
            % In our case the constraint is Cx >= 0 so the hessian is 0
            Hc = sparse(self.objSize, self.objSize);
        end
        
        function H = hlag_local(self, x, ~)
            %% Computes the hessian of the lagrangian
            % Considering all constraints are linear, the hessian of the
            % lagrangian is the hessian of the objective function
            H = self.hobj_local(x);
        end
        
        function w = hlagprod_local(self, x, ~, v)
            %% Computes the hessian of the lagrangian times vector
            % Considering all constraints are linear, the hessian of the
            % lagrangian is the hessian of the objective function
            
            % hobj_local(x) returns a Spot operator representing hess(x)
            w = self.hobjprod_local(x, [], v);
        end
        
        function w = hconprod_local(self, ~, ~, ~)
            %% Computes the hessian of the constraints times vector
            w = sparse(self.objSize, 1);
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
