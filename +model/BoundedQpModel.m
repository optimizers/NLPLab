classdef BoundedQpModel < model.QpModel
    %% BoundedQpModel
    % Subclass of QpModel with a "project" function that projects on the
    % bounds.
    
    
    methods (Access = public)
        
        function self = BoundedQpModel(name, x0, bL, bU, c, Q)
            A = [];
            n = length(bL);
            cL = -Inf(n, 1);
            cU = Inf(n, 1);
            self = self@model.QpModel(name, x0, cL, cU, bL, bU, A, c, Q);
        end
        
        function z = project(self, x)
           z = min(max(x, self.bL), self.bU); 
        end
        
    end
    
end
