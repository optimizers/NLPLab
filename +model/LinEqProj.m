classdef LinEqProj < handle
    %% LinEqProj
    % Class that provides a projection function on the set
    %
    %   { z | C * z = c }
    %
    % by using the analytical solution to
    %
    %   min_z { 1/2 || z - x ||^2 : C * z = c }
    
    methods (Access = public)
        
        function z = project(self, x)
            %% Project on linear equalities
            % Analytical solution of
            % min_z { 1/2 || z - x ||^2 : C * z = c }
            % Input:
            %   - x: point to project
            % Output:
            %   - z: projection of x
            
            assert(all(self.cL == self.cU));
            
            C = self.C;
            z = (C * C') \ (self.cU - C*x);
            z = x + (C' * z);
        end
        
        function z = projectSel(self, x, sel)
            %% ProjectSel - project on lin. eq. for selected indices
            % Analytical solution of
            % min_z { 1/2 || z - x ||^2 : C * z = c }
            % Inputs
            %   - x: point to project
            %   - sel: bool array that represents a subset of x to project
            % Output
            %   - z: projection of x
            
            assert(all(self.cL == self.cU));
            
            xSel = x(sel);
            C = self.C(sel, :);
            c = self.cU(sel);
            
            z = (C * C') \ (c - C * xSel);
            z = xSel + (C' * z);
        end
        
    end
    
end