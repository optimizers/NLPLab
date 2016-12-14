classdef LinIneqProj < handle
    
    
    methods (Access = public)
        
        function z = project(self, x)
            %% Project on linear inequalities
            % Solves min_z { 1/2 * || z - x ||^2 : cL <= C*z <= cU }
            % using MATLAB's lsqlin
            C = speye(length(x));
            d = x;
            % lsqlin only allows to solve for A*x <= b, therefore the
            % lower bound must be rewritten
            A = [self.C; -self.C];
            b = [self.cU; -self.cL];
            options = optimoptions('lsqlin', 'Display', 'off', ...
                'Algorithm', 'interior-point', 'OptimalityTolerance', ...
                1e-12, 'StepTolerance', 1e-15);
            z = lsqlin(C, d, A, b, [], [], [], [], [], options);
        end
        
        function z = projectSel(self, x, sel)
            %% ProjectSel - project on lin. ineq. for selected indices
            % Solves
            % min_z { 1/2 * || z - x ||^2 : cL <= C*z <= cU }
            % using MATLAB's lsqlin
            
            C = speye(length(x(sel)));
            d = x(sel);
            % lsqlin only allows to solve for A*x <= b, therefore the
            % lower bound must be rewritten
            A = [self.C(sel); -self.C(sel)];
            b = [self.cU(sel); -self.cL(sel)];
            
            options = optimoptions('lsqlin', 'Display', 'off', ...
                'Algorithm', 'interior-point', ...
                'OptimalityTolerance', 1e-12, ...
                'StepTolerance', 1e-15);
            
            z = lsqlin(C, d, A, b, [], [], [], [], [], options);
        end
        
    end
    
end