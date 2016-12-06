classdef SlackModel < model.NlpModel
   % SLACKMODEL  Equality constraints with bounds.
   %
   % Derives the following slack formulation
   %
   % minimize  f(x,s)
   % subj to   c(x) - s = 0
   %           bL <= x <= bU
   %           cL <= s <= cU
   %
   % from the inequality-based formulation:
   %
   % minimize  f(x)
   % subj to   cL <= c(x) <= cU
   %           bL <=   x  <= bU
   %
   % Thus, the problem appears as such:
   %
   % minimize  f(x)  subj to  c(x) = 0, bL <= x <= bU.
   
   properties
      nlp    % original inequality-based object
      islack % indictor of slack variables
      n_I    % number of inequality constraints
      c_I    % inequality constraints
   end

   properties (SetAccess = private, Hidden = false)
      Aeq    % Jacobian of linear constraints
      beq    % RHS of linear constraints
   end
   
   methods
      
      function self = SlackModel(nlp)

         constraints_I = find(nlp.cL ~= nlp.cU);
         nI = size(constraints_I,1);
         
         % Upper and lower bounds for the variables and slacks.
         bL = [ nlp.bL; nlp.cL(constraints_I) ];
         bU = [ nlp.bU; nlp.cU(constraints_I) ];

         % The linear and nonlinear constraints are equalities, ie,
         % 0 <= c(x) - s <= 0.
         cL = nlp.cL;
         cU = nlp.cU;
         cL(constraints_I) = zeros(nI, 1);
         cU(constraints_I) = zeros(nI, 1);

         % Initial point. Set slacks to be feasible.
         c = nlp.fcon(nlp.x0);
         x0 = [ nlp.x0; c(constraints_I) ];

         % Instantiate from the base class.
         self = self@model.NlpModel(nlp.name, x0, cL, cU, bL, bU);

         % Identify the linear constraints.
         self.linear = nlp.linear;
         
         % Create an indetifier for slack variables.
         self.islack = [ false(nlp.n,1); true(nI,1) ];   

         % Jacobian sparsity pattern of the slack model.
         Jx = nlp.gcon(nlp.x0);
         J = [spones(Jx) sparse(nlp.m, nI)];
         J(constraints_I , nlp.n +1: nlp.n + nI) = speye(nI);
         self.Jpattern = J;

         % Hessian sparsity pattern.
         y = ones(size(c));
         HL = nlp.hlag(nlp.x0, y);
         nS = self.n - nlp.n;
         self.Hpattern = [ spones(HL)         sparse(nlp.n, nS)
                           sparse(nS, nlp.n)  sparse(nS   , nS) ];
         
         % Store linear Jacobian and RHS.
         self.Aeq =  J(self.linear, :);
         self.beq = cL(self.linear, :);
         
         % Store the original NLP model.
         self.nlp = nlp;
         
         % Store inequality stuff
         self.n_I = nI;
         self.c_I = constraints_I;
         
      end

      function f = fobj_local(self, xs)
         x = xs(~self.islack,:);
         f = self.nlp.fobj(x);
      end
      
      function g = gobj_local(self, xs)
         x = xs(~self.islack,:);
         gx = self.nlp.gobj(x);
         g = [gx; sparse(self.n_I, 1)];
      end
      
      function H = hobj_local(self, xs)
         x = xs(~self.islack,:);
         Hx = self.nlp.hobj(x);
         nmZ = sparse(self.nlp.n, self.n_I);
         mmZ = sparse(self.n_I, self.n_I);
         H = [ Hx     nmZ
               nmZ'   mmZ ];
      end
      
      function c = fcon_local(self, xs)
         x = xs(~self.islack,:);
         s = xs( self.islack,:);
         c = self.nlp.fcon(x) ;
         c(self.c_I) = c(self.c_I) - s;
      end
      
      function J = gcon_local(self, xs)
         x = xs(~self.islack,:);
         Jx = self.nlp.gcon(x);
         J = [Jx sparse(self.nlp.m,self.n_I)];
         J(self.c_I , self.nlp.n +1: self.nlp.n + self.n_I) = -speye(self.n_I);
      end
      
      function HL = hlag_local(self, xs, y)
         x = xs(~self.islack,:);
         H = self.nlp.hlag(x, y);
         nmZ = sparse(self.nlp.n, self.nI);
         mmZ = sparse(self.nI,self.nI);
         HL = [ H     nmZ
                nmZ'  mmZ ];
      end
      
      function Hv = hlagprod_local(self, xs, y, vv)
         x = xs(~self.islack);
         v = vv(~self.islack);
         Hv = sparse(self.n, 1);
         Hv(~self.islack) = self.nlp.hlagprod(x, y, v);
      end
      
   end % methods
      
end % classdef
