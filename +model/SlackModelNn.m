classdef SlackModelNn < model.NlpModel
   % SLACKMODEL  Equality constraints with bounds.
   %
   % Derives the following slack formulation
   %
   % minimize f(x,s,sUx,sLx,sUc,sLc)
   % subj to  c(x) - s                         = 0  (m)
   %            x      + sUx                   = bU (nsUx)
   %            x            - sLx             = bL (nsLx)
   %                 s             + sUc       = cU (msUc)
   %                 s                   - sLc = cL (msLc)
   % with     (x, s) free; (sUx,sLx,sUc,sLc)  >= 0.
   %
   % from the inequality-based formulation
   %
   % minimize  f(x)
   % subj to   cL <= c(x) <= cU
   %           bL <=   x  <= bU
   %
   %
   properties (SetAccess = private, Hidden = true)
      J      % Jacobian
      indc
      indx
      inds   % Index structure to keep track of slacks
   end

   properties (SetAccess = private, Hidden = false)
      Aeq    % Jacobian of linear constraints
      beq    % RHS of linear constraints
   end
   
   properties
      nlp    % original inequality-based object
      indxs  % index structure      
   end
   
   methods
      
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      
      function self = SlackModelNn(nlp)

         m = nlp.m;
         n = nlp.n;

         % Slacks on x
         indxl.sUx = nlp.jUpp;
         indxl.sLx = nlp.jLow;
         sUx = nlp.bU(indxl.sUx) - nlp.x0(indxl.sUx);
         sLx = nlp.x0(indxl.sLx) - nlp.bL(indxl.sLx);

         % Slacks on c, and the slacks on the slacks!
         c = nlp.fcon(nlp.x0);
         indcl.s = ~nlp.iFix;
         s = c(indcl.s);

         indcl.sUc = ~nlp.iFix & nlp.iUpp;
         indsl.sUc = indcl.sUc(~nlp.iFix);         
         sUc = nlp.cU(indsl.sUc) - c(indsl.sUc);
         
         indcl.sLc = ~nlp.iFix & nlp.iLow;
         indsl.sLc = indcl.sLc(~nlp.iFix);
         sLc = c(indsl.sLc) - nlp.cL(indsl.sLc);

         x0 = [ nlp.x0; s; sUx; sLx; sUc; sLc ];

         % Create an indentifier for slack variables.
         sz.x   = n;
         sz.s   = length(s  );
         sz.sUx = length(sUx);
         sz.sLx = length(sLx);
         sz.sUc = length(sUc);
         sz.sLc = length(sLc);
         indxsl = model.buildind(sz);

         % Upper and lower bounds for the variables and slacks.
         %    [  x   s   sUx  sLx  sUc  sLc ]
         bL = zeros(size(x0));
         bL(indxsl.x) = -inf;
         bL(indxsl.s) = -inf;
         bU = inf(size(x0));
         
         % If the original problem has nonnegative lower bounds, it can't
         % hurt to explicitly enforce a nonnegative constraint on those
         % bounds (for example, if the objective involves functions that
         % are not defined for negative quantities).
         % TODO: reconsider this. Can be useful to keep vars unconstrained.
         % bL(nlp.bL >= 0) = 0;

         % The linear and nonlinear constraints are equalities.          
         cL = [ zeros(m,1);
                nlp.bU(indxl.sUx)
                nlp.bL(indxl.sLx)
                nlp.cU(indcl.sUc)
                nlp.cL(indcl.sLc) ];
         cU = cL;

         % Adjust the Jacobian sparsity pattern from the original model.
         %    x   s   sUx   sLx   sUc   sLc
         %  [ J   -I                        ] m   :  c - s
         %  [+I       +I                    ] nsUx:  x + sUx
         %  [+I             -I              ] nsLx:  x - sLx
         %  [     +I              +I        ] nsUc:  s + sUc
         %  [     +I                    -I  ] nsLc:  s - sLc
         %    n   ns  nsUx  nsLx  nsUc  nsLc
         ns   = sz.s;
         nsUx = sz.sUx;
         nsLx = sz.sLx;
         nsUc = sz.sUc;
         nsLc = sz.sLc;
         
         % --------------------------------------------------------------------
         % Instantiate from the base class.
         % --------------------------------------------------------------------
         self = self@model.NlpModel(nlp.name, x0, cL, cU, bL, bU);

         % --------------------------------------------------------------------
         % Identify the linear constraints.
         % --------------------------------------------------------------------
         self.linear = true(self.m,1);
         self.linear(1:m) = nlp.linear;
                
         % --------------------------------------------------------------------
         % Jacobian
         % --------------------------------------------------------------------
         Im = speye(m); In = speye(n);
         self.Jpattern = ...
         [ nlp.Jpattern        -Im(:,indcl.s)           sparse(m   ,nsUx)         sparse(m   ,nsLx)         sparse(m   ,nsUc)        sparse(m   ,nsLc)
             +In(indxl.sUx,:)   sparse(nsUx,ns)        +In(indxl.sUx,indxl.sUx)   sparse(nsUx,nsLx)         sparse(nsUx,nsUc)        sparse(nsUx,nsLc)
             +In(indxl.sLx,:)   sparse(nsLx,ns)         sparse(nsLx,nsUx)        -In(indxl.sLx,indxl.sLx)   sparse(nsLx,nsUc)        sparse(nsLx,nsLc)
           sparse(nsUc    ,n)  +Im(indcl.sUc,indcl.s)   sparse(nsUc,nsUx)         sparse(nsUc,nsLx)        +Im(indsl.sUc,indsl.sUc)  sparse(nsUc,nsLc)
           sparse(nsLc    ,n)  +Im(indcl.sLc,indcl.s)   sparse(nsLc,nsUx)         sparse(nsLc,nsLx)         sparse(nsLc,nsUc)       -Im(indsl.sLc,indsl.sLc) ];
         %  x(n)                s(ns)                   sUx(nsUx)

         % Store a copy of the Jacobian. Useful later in Jacobian evaluation.
         % May as well make the original block correct.
         self.J = self.Jpattern;
         self.J(1:m,1:n) = nlp.gcon(nlp.x0);
         
         % Store linear Jacobian.
         self.Aeq = self.J(self.linear, :);
         
         % --------------------------------------------------------------------
         % Linear RHS.
         % Seperately consider original 1:m constraints that are fixed and
         % not fixed, ie,
         % c_1(x) := c_10(x) + const_1         = RHS_1  (no change to RHS)
         % c_2(x) := c_20(x) + const_2 - slack = 0      (zero RHS)
         % --------------------------------------------------------------------
         
         % Figure out the constant terms for both.
         c0 = nlp.fcon(zeros(nlp.n,1));

         RHS = cL;
         ic = false(size(self.linear));

         % RHS for original linear and fixed.
         ic(1:m) = nlp.linear &  nlp.iFix;
         RHS(ic) = nlp.cL(ic(1:m)) - c0(ic(1:m));
         
         % RHS for original linear and NOT fixed.
         ic(1:m) = nlp.linear & ~nlp.iFix;
         RHS(ic) =                 - c0(ic(1:m));
         
         % Set the RHS for the linear constraints.
         self.beq = RHS(self.linear);
                  
         % --------------------------------------------------------------------
         % Hessian pattern.
         % --------------------------------------------------------------------
         y = ones(size(c));
         HL = nlp.hlag(nlp.x0, y);
         nS = self.n - n;
         self.Hpattern = [ spones(HL)     sparse(n, nS)
                           sparse(nS, n)  sparse(nS, nS) ];
         
         % --------------------------------------------------------------------
         % Store various things.
         % --------------------------------------------------------------------
         self.nlp = nlp;
         self.indxs = indxsl;
         self.indc  = indcl;
         self.indx  = indxl;
         self.inds  = indsl;
      end
            
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      function f = fobj_local(self, xs)
         %FOBJ  Objective function.
         x = xs(self.indxs.x,:);
         f = self.nlp.fobj(x);
      end

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      function gxs = gobj_local(self, xs)
         %HOBJ  Gradient of objective function.
         % gxs = [ g ]  x   n original gradient
         %       [ 0 ]  s   m
         %       [ 0 ]  sUx n
         %       [ 0 ]  sLx n
         %       [ 0 ]  sUc m
         %       [ 0 ]  sLc m         
         x = xs(self.indxs.x,:);
         gx = self.nlp.gobj(x);
         gxs = zeros(self.n, 1);
         gxs(self.indxs.x) = gx;
      end
         
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      
      function Hxs = hobj_local(self, xs)
         %HOBJ  Hessian of objective function.
         x = xs(self.indxs.x,:);
         Hx = self.nlp.hobj(x);
         Hxs = sparse([],[],[],self.n, self.n,nnz(Hx));
         Hxs(self.indxs.x, self.indxs.x) = Hx;
      end

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      function cxs = fcon_local(self, xs)
         %FCON  Constraint function.
         x = xs(self.indxs.x,:);
         s = xs(self.indxs.s,:);
         cx = self.nlp.fcon(x);
         % c(x) - s                         = 0  (m)
         %   x      + sUx                   = bU (n)
         %   x            - sLx             = bL (n)
         %        s             + sUc       = cU (m)
         %        s                   - sLc = cL (m)
         cx( self.indc.s,:) = cx( self.indc.s,:) - s;
         cx(~self.indc.s,:) = cx(~self.indc.s,:) - self.nlp.cL(~self.indc.s,:);
         cxs = [ cx
                 x(self.indx.sUx,:) + xs(self.indxs.sUx,:)
                 x(self.indx.sLx,:) - xs(self.indxs.sLx,:)
                 s(self.inds.sUc,:) + xs(self.indxs.sUc,:)
                 s(self.inds.sLc,:) - xs(self.indxs.sLc,:) ];
      end
      
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      function J = gcon_local(self, xs)
         %GCON  Constraint Jacobian.
         x = xs(self.indxs.x,:);
         J = self.J;
         Jx = self.nlp.gcon(x);
         J(1:self.nlp.m,self.indxs.x) = Jx;
      end
      
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
      function HL = hlag_local(self, xs, yy)
         %HLAG  Hessian of Lagrangian (sparse matrix).
         x = xs(self.indxs.x,:);
         y = yy(1:self.nlp.m,:);
         H = self.nlp.hlag(x, y);
         n = self.nlp.n;
         nS = self.n - n;
         HL = [ H              sparse(n, nS)
                sparse(nS, n)  sparse(nS, nS) ];
      end
      
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
      function Hv = hlagprod_local(self, xs, yy, vv)
         %HLAGPROD  Hessian-vector product with Hessian of Lagrangian.
         x = xs(self.indxs.x,:);
         y = yy(1:self.nlp.m,:);
         v = vv(self.indxs.x,:);
         Hv = zeros(self.n, 1);
         Hv(self.indxs.x) = self.nlp.hlagprod(x, y, v);
      end
      
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      
      function HC = hcon_local(self, xs, yy)
         %HCON  Hessian of Lagrangian (without objective; sparse matrix).
         x = xs(self.indxs.x,:);
         y = yy(1:self.nlp.m,:);
         H = self.nlp.hcon(x, y);
         n = self.nlp.n;
         nS = self.n - n;
         HC = [ H              sparse(n, nS)
                sparse(nS, n)  sparse(nS, nS) ];
      end
            
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      function Hv = hconprod_local(self, xs, yy, vv)
         %HCONPROD  Hessian-vector product with HCON.
         x = xs(self.indxs.x,:);
         y = yy(1:self.nlp.m,:);
         v = vv(self.indxs.x,:);
         Hv = zeros(self.n, 1);
         Hv(self.indxs.x) = self.nlp.hconprod(x, y, v);
      end
      
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      
      function z = ghivprod_local(self, xs, gxs, vxs)
         x = xs (self.indxs.x,:);
         g = gxs(self.indxs.x,:);
         v = vxs(self.indxs.x,:);
         z = zeros(self.m,1);
         z(~self.linear) = self.nlp.ghivprod(x, g, v);
      end
      
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   end % protected methods

end % classdef
