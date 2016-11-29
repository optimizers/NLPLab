classdef lpmodel_spot < model.nlpmodel
    % nlpmodel where 
    %   objectif is linear     : f(x) = c' * x
    %   constraints are linear : c(x) = A  * x
    
   properties (SetAccess = private, Hidden = false)
      A    % Jacobian of linear constraints (spot operator)
      c    % Gradient of linear objectif
   end
    
    methods (Sealed = true)
        
        function o = lpmodel_spot(name, x0, cL, cU, bL, bU, A, c)
            
            o = o@model.nlpmodel(name, x0, cL, cU, bL, bU);
            
            % Store other things.
            o.A = A;
            o.c = c;
            
            % In lpmodel, all constraints are linear
            o.linear = true(o.m, 1);
        end
        
        function f = fobj_local(self, x)
            f = self.c' * x;
        end
        
        function g = gobj_local(self, ~)
            g = self.c;
        end
        
        function H = hobj_local(self, ~)
            H = zeros(self.n);
        end
        
        function c = fcon_local(self, x)
            c = self.A * x;
        end
        
        function J = gcon_local(self, ~)
            J = self.A;
        end
        
        function Hc = hcon_local(self, ~, ~)
            Hc = zeros(self.n);
        end
        
        function H = hlag_local(self, ~, ~)
            H = zeros(self.n);
        end
        
        function w = hlagprod_local(self, ~, ~, ~)
            w = zeros(self.n, 1);
        end
        
        function w = hconprod_local(self, ~, ~, ~)
            w = zeros(self.n, 1);
        end
        
        function w = ghivprod_local(self, ~, ~, ~)
            w = zeros(self.n, 1);
        end
    end
end
