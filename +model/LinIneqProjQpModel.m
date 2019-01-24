classdef LinIneqProjQpModel < model.QpModel & model.LinIneqProj
    methods (Access = public)
        function self = LinIneqProjQpModel(varargin)
            self = self@model.QpModel(varargin{:});
        end
    end
end
