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
        
        function [z, solved] = project(self, x)
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
            solved = true;
        end
        
        function [z, solved] = projectSel(self, x, sel)
            %% ProjectSel - project on lin. eq. for selected indices
            % Analytical solution of
            % min_z { 1/2 || z - x ||^2 : C * z = c }
            % Inputs
            %   - x: point to project
            %   - sel: bool array that represents a subset of 
            %   the constraints to project on
            % Output
            %   - z: projection of x
            
            assert(all(self.cL == self.cU));
            
            C = self.C(sel, :);
            c = self.cU(sel);
            
            z = (C * C') \ (c - C * x);
            z = x + (C' * z);
            solved = true;
        end
        
    end
    
end