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
            1e-12);
    end
    
    
    methods (Access = public)
        
        function z = project(self, x)
            %% Project on linear inequalities
            % Solves min_x { 1/2 * || x - xbar ||^2 : cL <= C * x <= cU }
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
        
        function z = projectSel(self, x, sel)
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
            A = [self.C(sel); -self.C(sel)];
            b = [self.cU(sel); -self.cL(sel)];
            
            z = lsqlin(C, d, A, b, [], [], [], [], [], self.LSQLIN_OPTS);
        end
        
    end
    
end