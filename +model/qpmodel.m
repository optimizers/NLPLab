classdef qpmodel < model.nlpmodel
    % nlpmodel where 
    %   objectif is quadratic     : f(x) = c' * x + 1/2 x' * Q * x
    %   constraints are linear    : c(x) = A  * x
    
   properties (SetAccess = private, Hidden = false)
      A    % Jacobian of linear constraints
      c    % Gradient of linear objectif
      Q    % Hessien (symmetric)
   end
    
    methods (Sealed = true)
        
        function o = qpmodel(name, x0, cL, cU, bL, bU, A, c, Q)
            
            o = o@model.nlpmodel(name, x0, cL, cU, bL, bU);
            
            % Store other things.
            o.A = A;
            o.c = c;
            o.Q = Q;
            
            % In lpmodel, all constraints are linear
            o.linear = true(o.m, 1);
        end
        
        function f = fobj_local(self, x)
            f = self.c' * x + 0.5 * x' * self.Q * x;
        end
        
        function g = gobj_local(self, x)
            g = self.c + self.Q * x;
        end
        
        function H = hobj_local(self, ~)
            H = self.Q;
        end
        
        function c = fcon_local(self, x)
            c = self.A * x;
        end
        
        function J = gcon_local(self, ~)
            J = self.A;
        end
        
        function Hc = hcon_local(self, ~, ~)
            Hc = sparse(self.n, self.n);
        end
        
        function H = hlag_local(self, ~, ~)
            % Why??? This seems wrong...
%             H = sparse(self.n, self.n);
            H = self.Q;
        end
        
        function w = hlagprod_local(self, ~, ~, v)
            % Why?? This seems wrong...
%             w = sparse(self.n, 1);
            w = self.Q * v;
        end
        
        function w = hconprod_local(self, ~, ~, ~)
            w = sparse(self.n, 1);
        end
        
        function w = ghivprod_local(self, ~, ~, ~)
            w = sparse(self.n, 1);
        end
    end
end
