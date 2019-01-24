classdef LinIneqProjAmplModel < model.AmplModel & model.LinIneqProj
    methods (Access = public)
        
        function self = LinIneqProjAmplModel(varargin)
            self = self@model.AmplModel(varargin{:});
            warning('Modified fcon & gcon for bounded CUTE/COPS problems!');
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
        
    end
end
