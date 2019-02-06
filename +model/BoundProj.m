classdef BoundProj < handle
    %% BoundProj
    % Class that provides a projection function on the set
    %
    %   { z | bL <= z <= bU }
    %
    % by using the analytical solution to
    %
    %   min_x { 1/2 || z - x ||^2 : bL <= z <= bU }
    
    properties (SetAccess = private)
       solved = true; 
    end
    
    methods (Access = public)
        
        function [z, solved] = project(self, x)
            %% Project on bounds
            % Input
            %   - x: point to project
            % Output
            %   - z: projection of x
            z = min(max(x, self.bL), self.bU);
            solved = true;
        end
        
        function [z, solved] = projectSel(self, x, sel)
            %% ProjectSel - project on bounds for a selected set of indices
            % Inputs
            %   - x: point to project
            %   - sel: bool array that represents a subset of x to project
            % Output
            %   - z: projection of x
            z = min(max(x(sel), self.bL(sel)), self.bU(sel));
            solved = true;
        end
        
    end
    
end