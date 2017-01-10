classdef LinEqProj < handle
    
    
    methods (Access = public)
        
        function z = project(self, x)
            %% Project on linear equalities
            % Analytical solution of
            % min_z     1/2 * || z - x ||^2
            %           C*z = 0
            error('GENERALIZE ME!');
            C = self.C;
            w = (C * C'') \ (-C*x);
            z = x + (C'' * w);
        end
        
        function z = projectSel(self, x, sel)
            %% ProjectSel - project on lin. eq. for selected indices
            % Analytical solution of
            % min_z     1/2 * || z - x ||^2
            %           C*z = 0
            error('GENERALIZE ME!');
            
            xSel = x(sel);
            C = self.C(sel, :);
            
            w = (C * C') \ (-C * xSel);
            z = xSel + (C'' * w);
        end
        
    end
    
end