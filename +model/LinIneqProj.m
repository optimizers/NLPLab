classdef LinIneqProj < handle
    %% LinIneqProj
    % Class that provides a projection function on the set
    %
    %   { z | cL <= C * z <= cU }
    %
    % by solving
    %
    %   min_x { 1/2 || z - x ||^2 : cL <= C * z <= cU }
    
    properties (SetAccess = private)
       solved = true; 
    end
    
    properties (Constant, Hidden = false)
        % MATLAB's lsqlin parameters
        LSQLIN_OPTS = optimoptions('lsqlin', 'Display', 'off', ...
            'Algorithm', 'interior-point', 'OptimalityTolerance', ...
            1e-12, 'MaxIterations', 1e5);
        QUADPROG_OPTS = optimoptions('quadprog', 'Display', 'off', ...
            'Algorithm', 'interior-point-convex', ...
            'OptimalityTolerance', 1e-12, 'StepTolerance', 1e-12, ...
            'MaxIterations', 1e5);
        MINRES_OPTS = struct('rtol', 1e-12, 'etol', 1e-12, 'shift', 0, ...
            'show', false, 'check', true);
    end
    
    
    methods (Access = public)
        
        function z = project(self, x)
            %% Project on linear inequalities
            % Solves the dual of
            % min_x { 1/2 * || z - x ||^2 : cL <= C * z <= cU }
            % <=>
            % min_d { 1/2 * d' * A' * A * d + (A'*x + b)' * d : d >= 0 }
            % where
            %   d := [z; w]
            %   A := [-C', C']
            %   b := [cL; -cU]
            % using MATLAB's quadprog
            % Input
            %   - x: point to project
            % Output
            %   - z: projection of x
            
            A = [self.C', -self.C'];
            b = [-self.cL; self.cU];
            
            temp = sum(self.iUpp + self.iLow);
            
            lb = zeros(temp, 1);
            ub = inf(temp, 1);
            
            % H := A' * A
            % f := -A'*x + b
            d = quadprog(A'*A, A'*x + b, [], [], [], [], lb, ub, [], ...
                self.QUADPROG_OPTS);
            
            z = x + A * d;
        end
        
        function z = projectSel(self, x, sel)
            %% Project on linear inequalities
            % Solves the dual of
            % min_x { 1/2 * || z - x ||^2 : cL <= C * z <= cU }
            % <=>
            % min_d { 1/2 * d' * A' * A * d - (A'*x - b)' * d : d >= 0 }
            % where
            %   d := [z; w]
            %   A := [-C', C']
            %   b := [cL; cU]
            % using MATLAB's quadprog
            % Input
            %   - x: point to project
            % Output
            %   - z: projection of x
            
            xSel = x(sel);
            C = self.C(sel, :);
            
            A = [-C', C'];
            b = [self.cL(sel); -self.cU(sel)];
            
            temp = sum(self.iUpp(sel) + self.iLow(sel));
            
            lb = zeros(temp, 1);
            ub = inf(temp, 1);
            
            % H := A' * A
            % f := A'*x - b
            z = quadprog(A'*A, (A'*xSel + b), [], [], [], [], lb, ub, ...
                [], self.QUADPROG_OPTS);
            
            z = xSel + A * z;
        end
        
        function z = primalProject(self, x)
            %% Project on linear inequalities
            % Solves min_x { 1/2 * || z - x ||^2 : cL <= C * z <= cU }
            % using MATLAB's lsqlin
            % Input
            %   - x: point to project
            % Output
            %   - z: projection of x
            
            C = speye(length(x));
            d = x;
            
            % lsqlin only allows to solve for A*x <= b, therefore the
            % lower bound must be rewritten
            A = [self.C; -self.C];
            b = [self.cU; -self.cL];
            
            z = lsqlin(C, d, A, b, [], [], [], [], [], self.LSQLIN_OPTS);
        end
        
        function z = primalProjectSel(self, x, sel)
            %% ProjectSel - project on lin. ineq. for selected indices
            % Solves
            % min_x { 1/2 * || z - x ||^2 : cL <= C * z <= cU }
            % using MATLAB's lsqlin
            % Inputs
            %   - x: point to project
            %   - sel: bool array that represents a subset of x to project
            % Output
            %   - z: projection of x
            
            C = speye(length(x(sel)));
            d = x(sel);
            
            % lsqlin only allows to solve for A*x <= b, therefore the
            % lower bound must be rewritten
            A = [self.C(sel, :); -self.C(sel, :)];
            b = [self.cU(sel); -self.cL(sel)];
            
            z = lsqlin(C, d, A, b, [], [], [], [], [], self.LSQLIN_OPTS);
        end
        
        function nrmJac = normJac(self)
            nrmJac = norm(self.C);
        end
        
        %% This method doesn't correspond to the same problem
        % Solves the projection { x | C*x = 0 } for the primal variable
        function z = eqProject(self, d, fixed)
            %% EqProject - project vector d on equality constraints
            % Solves the problem
            % min   1/2 || w - d||^2
            %   w   sc (C*w)_i = 0, for i \not \in the working set
            % where C is the jacobian of the linear constraints and i
            % denotes the indices of the fixed variables. This problem has
            % an analytical solution.
            %
            % From the first order KKT conditions, we can obtain the
            % following set of equations:
            %
            %   w               = d + B*C*z
            %   (B*C*C'*B') * z = -B*C*d
            %
            % where z is the lagrange mutliplier associated with the
            % equality constraint.
            %
            % Inputs:
            %   - d: vector to project on the equality constraint set
            %   - fixed: indices not in the working set
            % Ouput:
            %   - v: projected direction
            
            % If B is the mask matrix such that
            % B := {b_i' = ith col of I for all i \in ~working}
            % Building reduced operators from object's attributes
            subC = self.C(fixed, :); % B * C
            % Using CG to solve (B*C*C'*B') z = -B*C*d
            [z, ~] = pcg(subC * subC', subC*(-d), 1e-10, 1e4);
            % For the unconstrained case, the solution is trivial
            z = d + (subC' * z);
        end
        
        function d = minEqProject(self, g, H, fixed)
            %% MinEqProject - Minimize the quadratic under linear eq.
            % Solves
            %    min_d { d'*H*d/2 + g'*d : B*C*d = 0 }
            % by solving the following linear system using minres_spot
            % [ H, (B*C)'; B*C, 0] * [d; -z] = [-g; 0]
            
            nFix = sum(fixed);
            subC = self.C(fixed, :);
            
            d = minres_spot([H, subC'; subC, zeros(nFix, nFix)], ...
                [-g; zeros(nFix, 1)], self.MINRES_OPTS);
            
            d = d(1 : end - nFix);
        end
        
        function d = minEqProject2(self, g, H, fixed)
            %% MinEqProject - Minimize the quadratic under linear eq.
            % Solves
            %    min_d { d'*H*d/2 + g'*d : B*C*d = 0 }
            
            beq = zeros(sum(fixed), 1);
            subC = self.C(fixed, :);
            
            d = quadprog(H, g, [], [], subC, beq, [], [], [], ...
                self.QUADPROG_OPTS);
        end
    end
    
end