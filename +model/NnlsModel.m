classdef NnlsModel < model.NlpModel
    %% Non-Negative Least Squares model
    % min_x { 1/2 * || A*x - b ||^2 : x >= 0 }
    
    
    %% Properties
    properties (SetAccess = private, Hidden = false)
        A;
        b;
    end
    
    
    %% Public methods
    methods (Access = public)
        
        function self = NnlsModel(A, b, x0, name)
            n = size(A, 2);
            if nargin < 3
                x0 = zeros(n, 1);
                name = '';
            elseif nargin < 4
                name = '';
            end
            
            bL = zeros(n, 1);
            bU = Inf(n, 1);
            cU = bU;
            cL = -bU;
            self = self@model.NlpModel(name, x0, cL, cU, bL, bU);
            self.A = A;
            self.b = b;
        end
        
        function [fObj, grad, hess] = obj(self, z)
            %% Function returning the obj. func, grad. and hess.
            fObj = self.fobj(z);
            if nargout > 1
                grad = self.gobj(z);
            end
            if nargout > 2
                hess = self.hobj(z);
            end
        end
        
        function f = fobj_local(self, x)
            f = 1/2 * norm(self.A*x - self.b);
        end
        
        function g = gobj_local(self, x)
            g = self.A' * (self.A*x - self.b);
        end
        
        function H = hobj_local(self, ~)
            H = self.A' * self.A;
        end
        
        function Atb = getAtb(self)
            Atb = self.A' * self.b;
        end
        
    end
    
end
