classdef slackmodel_spot < model.nlpmodel
    % sous entend que le nlpmodel d entr�e � un gradient des contraintes
    % de type opSpot

    properties
      nlp    % original inequality-based object
      islack % indictor of slack variables
      n_I    % number of inequality constraints
      c_I    % inequality constraints
    end

   methods

      function self = slackmodel_spot(nlp)

         constraints_I = find(nlp.cL ~= nlp.cU);
         nI = size(constraints_I, 1);

         % Upper and lower bounds for the variables and slacks.
         bL = [ nlp.bL; nlp.cL(constraints_I) ];
         bU = [ nlp.bU; nlp.cU(constraints_I) ];

         % The linear and nonlinear constraints are equalities, ie,
         % 0 <= c(x) - s <= 0.
         cL = nlp.cL;
         cU = nlp.cU;
         cL(constraints_I) = spalloc(nI, 1, nI);
         cU(constraints_I) = spalloc(nI, 1, nI);

         % Initial point. Set slacks to be feasible.
         c = nlp.fcon(nlp.x0);
         x0 = [ nlp.x0; c(constraints_I) ];

         % Instantiate from the base class.
         self = self@model.nlpmodel(nlp.name, x0, cL, cU, bL, bU);

         % Identify the linear constraints.
         self.linear = nlp.linear;

         % Create an identifier for slack variables.
         self.islack = [ false(nlp.n, 1); true(nI, 1) ];

         % Store the original NLP model.
         self.nlp = nlp;

         % Store inequality stuff
         self.n_I = nI;
         self.c_I = constraints_I;

      end
      
      % --
      function [f, g, H] = obj(self, xs)
          %% Evaluate obj. func., gradient and hessian of original model
          % Call obj on the model that has been slacked first, that way the
          % function can be redefined. If it's not, it's still going to end
          % up to nlpmodel's obj function, assuming self.nlp comes from a
          % subclass of nlpmodel.
          if nargout == 1
              % Nothing special to do, call slack fobj_local
              f = self.fobj_local(xs);
          elseif nargout == 2
              % Passing only the non-slack variables to local
              x = xs(~self.islack, :);
              % Calling slacked model's obj function
              [f, g] = self.nlp.obj(x);
              % Padding g
              g = [g; zeros(self.n_I, 1)];
          else % nargout == 3
              % Passing only the non-slack variables to local
              x = xs(~self.islack, :);
              % Calling slacked model's obj function
              [f, g, H] = self.nlp.obj(x);
              % Padding g
              g = [g; zeros(self.n_I, 1)];
              % Padding H
              nmZ = sparse(self.nlp.n, self.n_I);
              mmZ = sparse(self.n_I, self.n_I);
              H = [ H     nmZ
                  nmZ'   mmZ ];
          end
      end
        % --
      
      function y = op_jacobian(self, x, mode, op)
          if mode == 1
              y = op * x(1:self.nlp.n);
              y(self.c_I) = y(self.c_I) - x(self.nlp.n + 1 : self.nlp.n + self.n_I);
          else
              y = [op'*x; -x(self.c_I)];
          end
      end

      function f = fobj_local(self, xs)
         x = xs(~self.islack, :);
         f = self.nlp.fobj(x);
      end

      function g = gobj_local(self, xs)
         x = xs(~self.islack, :);
         gx = self.nlp.gobj(x);
         g = [gx; zeros(self.n_I, 1)];
      end

      function H = hobj_local(self, xs)
         x = xs(~self.islack, :);
         Hx = self.nlp.hobj(x);
         nmZ = sparse(self.nlp.n, self.n_I);
         mmZ = sparse(self.n_I, self.n_I);
         H = [ Hx     nmZ
               nmZ'   mmZ ];
      end

      function c = fcon_local(self, xs)
         x = xs(~self.islack, :);
         s = xs( self.islack, :);
         c = self.nlp.fcon(x) ;
         c(self.c_I) = c(self.c_I) - s;
      end

      function J = gcon_local(self, xs)
         x = xs(~self.islack, :);
         Jx = self.nlp.gcon(x);
         J = @(x, mode) op_jacobian(self, x, mode, Jx);
         J = opFunction(self.m, self.n, J);
      end

      function HL = hlag_local(self, xs, y)
         x = xs(~self.islack, :);
         H = self.nlp.hlag(x, y);
         nmZ = zeros(self.nlp.n, self.n_I);
         mmZ = zeros(self.n_I);
         HL = [ H     nmZ
                nmZ'  mmZ ];
      end

      function Hv = hlagprod_local(self, xs, y, vv)
         x = xs(~self.islack);
         v = vv(~self.islack);
         Hv = zeros(self.n, 1);
         Hv(~self.islack) = self.nlp.hlagprod(x, y, v);
      end
   end % methods
end
