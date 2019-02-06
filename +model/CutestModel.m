classdef CutestModel < model.NlpModel
    
    properties
        sparse % Flag indicates whether the model is dense or sparse
    end
    
    methods
        
        function self = CutestModel(dirname, sparse)
            % CutestModel constructor
            
            % Construct handle to either sparse or dense interface.
            if nargin < 2 || isempty(sparse)
                sparse = true;
            end
            
            cd(dirname);
            
            % Terminate previous cutest session
            try
                p = cutest_setup();
            catch ME
                terminate_msg = 'cutest_setup: cutest_terminate must be called first';
                msg_len = length(terminate_msg);
                if strcmp(ME.message(1:msg_len), terminate_msg)
                    cutest_terminate
                    p = cutest_setup();
                else
                    rethrow(ME);
                end
            end
            x0 = p.x;
            p.bl(p.bl < -1e10) = -Inf;
            p.cl(p.cl < -1e10) = -Inf;
            p.bu(p.bu > 1e10) = Inf;
            p.cu(p.cu > 1e10) = Inf;
            
            % Instantiate the base class.
            self = self@model.NlpModel(p.name, x0, p.cl, p.cu, p.bl, p.bu);
            
            % Record sparsity flag
            self.sparse = sparse;
            
            % Evaluate constraint and Jacobian at x0.
            if self.m > 0
                if sparse
                    [c, J] = cutest_scons(x0);
                else
                    [c, J] = cutest_cons(x0);
                end
            else
                c = [];
                J = [];
            end
            
            % Jacobian sparsity pattern.
            if issparse(J)
                self.Jpattern = spones(J);
            else
                self.Jpattern = ones(size(J));
            end
            
            % Hessian sparsity pattern.
            y = ones(size(c));
            if sparse
                H = cutest_sphess(x0, y);
                self.Hpattern = spones(H);
            else
                H = cutest_hess(x0, y);
                self.Hpattern = ones(size(H));
            end
            
            % Categorize nonlinear constraints. Ampl orders nonlinear
            % constraints first.
            self.linear = p.linear;
        end
        
        function f = fobj_local(~, x)
            %FOBJ  Objective function.
            f = cutest_obj(x);
        end
        
        function g = gobj_local(~, x)
            %GOBJ  Gradient bjective function.
            g = cutest_grad(x);
        end
        
        function H = hobj_local(self, x)
            %HOBJ  Hessian of objective function.
            if self.sparse
                H = cutest_isphess(x, 0);
            else
                H = cutest_ihess(x, 0);
            end
        end
        
        function c = fcon_local(self, x)
            %FCON  Constraint functions, nonlinear followed by linear.
            if self.m > 0
                c = cutest_cons(x);
            else
                c = [];
            end
        end
        
        function J = gcon_local(self, x)
            %GCON  Constraint functions Jacobian.
            if self.m > 0
                if self.sparse
                    [~,J] = cutest_scons(x);
                else
                    [~,J] = cutest_cons(x);
                end
            else
                J = zeros(0, self.n);
            end
        end
        
        function [Jprod, Jtprod] = gconprod_local(self, x)
            if self.m > 0
                Jprod = @(v) cutest_jprod(x,v);
                Jtprod = @(v) cutest_jtprod(x,v);
            else
                Jprod = @(v) zeros(0,1);
                Jtprod = @(v) zeros(self.n,1);
            end
        end
        
        function HC = hcon_local(self, x, y)
            if self.m > 0
                if self.sparse
                    HC = cutest_sphess(x, y);
                    HC = HC - self.hobj(x);
                else
                    HC = cutest_hess(x, y);
                    HC = HC - self.hobj(x);
                end
            else
                HC = sparse(self.n, self.n);
            end
        end
        
        function Hv = hconprod_local(self, x, y, v)
            if self.m > 0
                Hv = cutest_hprod(x, y, v);
                Hv = Hv - cutest_hprod(x, zeros(self.m,1), v);
            else
                Hv = zeros(self.n,1);
            end
        end
        
        function Hv = hlagprod_local(self, x, y, v)
            if self.m > 0
                Hv = cutest_hprod(x, y, v);
            else
                Hv = cutest_hprod(x, v);
            end
        end
        
        function HL = hlag_local(self, x, y)
            if self.m > 0
                if self.sparse
                    HL = cutest_sphess(x, y);
                else
                    HL = cutest_hess(x, y);
                end
            else
                if self.sparse
                    HL = cutest_sphess(x);
                else
                    HL = cutest_hess(x);
                end
            end
        end
        
        function gHiv = ghivprod_local(self, x, g, v)
            gHiv = zeros(self.m,1);
            for i=1:self.m
                gHiv(i) = g'*(cutest_isphess(x,i)*v);
            end
        end
    end
end
            