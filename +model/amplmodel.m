classdef amplmodel < model.nlpmodel
   
   properties (SetAccess = private, Hidden = true)
      ah              % ampl handle
   end

   properties
      sigma           % scale of the Lagrangian
      sparse          % flag indicates is dense of sparse model
   end

   methods

      function self = amplmodel(fname, sparse)
         %AMPLMODEL Constructor.
         
         % Construct handle to either sparse or dense interface.
         if nargin < 2 || isempty(sparse)
            sparse = false;
         end
         ahl = ampl.ampl_interface(fname, sparse);

         % Problem name.
         [~, name, ~] = fileparts(fname);
         
         % Number nonlinear constraints.
         nlc = ahl.nlc;

         % Instantiate the base class.
         self = self@model.nlpmodel(name, ahl.x0, ahl.cl, ahl.cu, ahl.bl, ahl.bu);
         
         % Record sparsity flag
         self.sparse = sparse;

         % Store the ampl handle.
         self.ah = ahl;
         
         % Evaluate constraint and Jacobian at x0.
         c = ahl.con(ahl.x0);
         J = ahl.jac(ahl.x0);
         
         % Jacobian sparsity pattern.
         if issparse(J)
            self.Jpattern = spones(J);
         else
            self.Jpattern = ones(size(J));
         end
         
         % Hessian sparsity pattern.
         y = zeros(size(c));
         H = ahl.hesslag(y);
         if issparse(H)
            self.Hpattern = spones(H);
         else
            self.Hpattern = ones(size(H));
         end

         % Categorize nonlinear constraints. Ampl orders nonlinear
         % constraints first.
         self.linear = true(self.m, 1);
         self.linear(1:nlc) = false;

         % Get the scale of the Lagrangian.
         self.sigma = ahl.sigma;

      end
      
      function self = lagscale(self, sigma)
         %LAGSCALE  Set the scale of the Lagrangian.
         %
         %LAGSCALE(SIGMA)  sets the Lagrangian to be
         %
         % L(x,y) = f(x) + sigma <c, y>.
         %
         % By default, sigma = -1.
         self.ah.lagscale(sigma);
         self.sigma = self.ah.sigma;
      end

      function write_sol(self, msg, x, y)
         self.ah.write_sol(msg, x, y);
      end

      function f = fobj_local(self, x)
         %FOBJ  Objective function.
         f = self.ah.obj(x);
      end

      function g = gobj_local(self, x)
         %GOBJ  Gradient bjective function.
         g = self.ah.grad(x);
      end
      
      function H = hobj_local(self, x)
         %HOBJ  Hessian of objective function.
         H = self.ah.hessobj(x);
      end
      
      function c = fcon_local(self, x)
         %FCON  Constraint functions, nonlinear followed by linear.
         c = self.ah.con(x);
      end
      
      function J = gcon_local(self, x)
         %GCON  Constraint functions Jacobian, nonlinear followed by linear.
         J = self.ah.jac(x);
      end
      
      function HC = hcon_local(self, x, y) %#ok<INUSL>
         HC = self.ah.hesscon(y);
      end

      function Hv = hconprod_local(self, x, y, v) %#ok<INUSL>
         Hv = self.ah.hessconprod(y, v);
      end

      function Hv = hlagprod_local(self, x, y, v) %#ok<INUSL>
         Hv = self.ah.hesslagprod(y, v);
      end

      function HL = hlag_local(self, x, y) %#ok<INUSL>
         HL = self.ah.hesslag(y);
      end
      
      function gHiv = ghivprod_local(self, x, g, v)
         gHiv = self.ah.ghivprod(x, g, v);
      end
      
   end

end % classdef
