classdef QpModel < model.NlpModel
    %% QpModel - general quadratic programming problem
    %   min_x   c' * x + 1/2 x' * Q * x
    %   sc      cL <= C * x <= cU
    %           bL <= x <= bU
    
    
    properties (SetAccess = private, Hidden = false)
        C    % Jacobian of linear constraints
        c    % Gradient of linear objective function
        Q    % Hessian (symmetric)
    end
    
    
    methods (Access = public, Sealed = true)
        
        function self = QpModel(Q, c, C, cL, cU, bL, bU, x0, name)
            %% Constructor
            
            % Setting optional parameters
            n = size(C, 2);
            if nargin < 8
                x0 = zeros(n, 1);
                name = '';
            elseif nargin < 9
                name = '';
            end
            
            self = self@model.NlpModel(name, x0, cL, cU, bL, bU);
            
            % Store other things.
            self.C = C;
            self.c = c;
            self.Q = Q;
            
            % In lpmodel, all constraints are linear
            self.linear = true(self.m, 1);
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
            f = self.c' * x + 0.5 * x' * self.Q * x;
        end
        
        function g = gobj_local(self, x)
            g = self.c + self.Q * x;
        end
        
        function [f, g] = fgobj_local(self, x)
            % If hessian product is costly, compute both f & g at the same
            % time
            v = self.Q * x;
            f = self.c' * x + 0.5 * x' * v;
            g = self.c + v;
        end
        
        function H = hobj_local(self, ~)
            H = self.Q;
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
            w = self.Q * v;
        end
        
        function w = hconprod_local(self, ~, ~, ~)
            w = sparse(self.n, 1);
        end

    end
    
end
