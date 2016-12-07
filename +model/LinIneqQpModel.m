classdef LinIneqQpModel < model.QpModel
    %% LinIneqQpModel
    % Subclass of QpModel with a "project" function that projects on the
    % linear inequalities.
    
    
    properties (Access = public)
        normJac;
    end
    
    methods (Access = public)
        
        function self = LinIneqQpModel(name, x0, A, cL, cU, c, Q)
            n = length(cL);
            % Unbounded problem
            bL = -Inf(n, 1);
            bU = Inf(n, 1);
            self = self@model.QpModel(name, x0, cL, cU, bL, bU, A, c, Q);
            self.normJac = norm(A);
        end
        
        function z = project(self, x)
            %% Project
            % Solves
            % min_z { 1/2 * || z - x ||^2 : cL <= A*z <= cU }
            % 0.5 * z' * z + 0.5 * x' * x - x' * z
            % using MATLAB's quadprog
            H = eye(length(x));
            f = -x;
            % Quadprog only allows to solve for A*x <= b, therefore the
            % lower bound must be rewritten
            A = [self.A; -self.A];
            b = [self.cU; -self.cL];
            
            options = optimoptions('quadprog', 'Display', 'off', ...
                'Algorithm', 'interior-point-convex', ...
                'OptimalityTolerance', 1e-12, ...
                'StepTolerance', 1e-15);
           
            z = quadprog(H, f, A, b, [], [], [], [], [], options);
        end
        
        function z = eqProject(self, x, ind)
            % Building reduced operators from object's attributes
            subA = self.A(ind, :); % B * A
            subAAt = subA*subA';
            
            w = subAAt \ (-subA*x);
            % For the unconstrained case, the solution is trivial
            z = x + (subA' * w);
        end
    end
    
end
