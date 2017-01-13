classdef LinIneqProj < handle
    %% LinIneqProj
    % Class that provides a projection function on the set
    %
    %   { z | cL <= C * z <= cU }
    %
    % by solving
    %
    %   min_x { 1/2 || z - x ||^2 : cL <= C * z <= cU }
    
    
    properties (Constant, Hidden = false)
        % MATLAB's lsqlin parameters
        LSQLIN_OPTS = optimoptions('lsqlin', 'Display', 'off', ...
            'Algorithm', 'interior-point', 'OptimalityTolerance', ...
            1e-12, 'MaxIterations', 1e5);
        QUADPROG_OPTS = optimoptions('quadprog', 'Display', 'off', ...
            'Algorithm', 'interior-point-convex', ...
            'OptimalityTolerance', 1e-12, 'StepTolerance', 1e-12, ...
            'MaxIterations', 1e5);
    end
    
    
    methods (Access = public)
        
        function z = dualProject(self, x)
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
            
            warning('This doesn''t seem correct.');
            
            A = [-self.C', self.C'];
            b = [self.cL; -self.cU];
            
            temp = sum(self.iUpp + self.iLow);
            
            lb = zeros(temp, 1);
            ub = inf(temp, 1);
            
            % H := A' * A
            % f := -A'*x + b
            d = quadprog(A'*A, A'*x + b, [], [], [], [], lb, ub, [], ...
                self.QUADPROG_OPTS);
            
            z = x + A * d;
        end
        
        function z = dualProjectSel(self, x, sel)
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
            
            warning('This doesn''t seem correct.');
            
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
        
        function z = project(self, x)
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
        
    end
    
end