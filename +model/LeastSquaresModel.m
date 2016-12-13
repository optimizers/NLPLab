classdef LeastSquaresModel < model.NlpModel
    %% Norm-2 least squares problem under linear constraints or bounds
    % min_x     1/2 * || A*x - b ||^2
    %           cL <= C * x <= cU
    %           bL <= x <= bU
    
    
    %% Properties
    properties (SetAccess = private, Hidden = false)
        A; % Coefficient matrix
        b; % Vector of measurements
        C; % Jacobian of linear constraints
    end
    
    
    %% Public methods
    methods (Access = public, Sealed = true)
        
        function self = LeastSquaresModel(A, b, C, cL, cU, bL, bU, x0, ...
                name)
            
            % Setting optional parameters
            n = size(A, 2);
            if nargin < 8
                x0 = zeros(n, 1);
                name = '';
            elseif nargin < 9
                name = '';
            end
            
            % Calling superclass constructor
            self = self@model.NlpModel(name, x0, cL, cU, bL, bU);
            
            % Storing other properties
            self.A = A;
            self.b = b;
            self.C = C;
        end
        
        function [fObj, grad, hess] = obj(self, x)
            %% Function returning the obj. func, grad. and hess.
            if nargout == 1
                fObj = self.fobj(x);
            elseif nargout == 2
                [fObj, grad] = self.fgobj(x);
            else % nargout == 3
                [fObj, grad] = self.fgobj(x);
                hess = self.hobj(x);
            end
        end
        
        function f = fobj_local(self, x)
            f = 1/2 * norm(self.A * x - self.b);
        end
        
        function g = gobj_local(self, x)
            g = self.A' * (self.A * x - self.b);
        end
        
        function [f, g] = fgobj_local(self, x)
            %% Compute both f & g at the same time in case it's expensive
            r = self.A*x - self.b;
            f = 1/2 * (r' * r);
            g = self.A' * r;
        end
        
        function H = hobj_local(self, ~)
            H = self.A' * self.A;
        end
        
        function c = fcon_local(self, x)
            c = self.C * x;
        end
        
        function J = gcon_local(self, ~)
            J = self.C;
        end
        
        function Hc = hcon_local(self, ~, ~)
            Hc = sparse(self.n, self.n);
        end
        
        function w = hobjprod_local(self, ~, ~, v)
            w = self.A' * self.A * v;
        end
        
        function w = hconprod_local(self, ~, ~, ~)
            w = sparse(self.n, 1);
        end
        
    end
    
end
