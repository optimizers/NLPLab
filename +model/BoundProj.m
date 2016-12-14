classdef BoundProj < handle
    
    
    methods (Access = public)
        
        function z = project(self, x)
            %% Project on bounds
            z = min(max(x, self.bL), self.bU);
        end
        
        function z = projectSel(self, x, sel)
            %% ProjectSel - project on bounds for a selected set of indices
            z = min(max(x(sel), self.bL(sel)), self.bU(sel));
        end
        
    end
    
end