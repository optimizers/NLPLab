classdef BoundProjAmplModel < model.AmplModel & model.BoundProj
    methods (Access = public)
        
        function self = BoundProjAmplModel(varargin)
            self = self@model.AmplModel(varargin{:});
            %% Modified fcon & gcon for bounded CUTE/COPS problems!
            self.cU = self.bU;
            self.cL = self.bL;
            self.m = self.n;
            self.iFix = self.jFix;
            self.iInf = self.jInf;
            self.iLow = self.jLow;
            self.iUpp = self.jUpp;
            self.iFre = self.jFre;
            self.iTwo = self.jTwo;
            self.Jpattern = speye(self.n);
        end
        
        %% THIS SHOULD BE REMOVED if not used on CUTE/COPS BOUNDED problems
        function c = fcon_local(~, x)
            c = x;
        end
        
        function J = gcon_local(self, ~)
            J = speye(self.n);
        end
        
        function nrmJac = normJac(~)
           nrmJac = 1; 
        end
        
        function x = projectMixed(self, x, fixed)
            x(~fixed) = self.projectSel(x, ~fixed);
        end
        
    end
end
