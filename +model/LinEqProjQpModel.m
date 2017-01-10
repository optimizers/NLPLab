classdef LinEqProjQpModel < model.QpModel & model.LinEqProj
    methods (Access = public)
        function self = LinEqProjQpModel(varargin)
            self = self@model.QpModel(varargin{:});
        end
    end
end
