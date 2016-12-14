classdef BoundProjQpModel < model.QpModel & model.BoundProj
    methods (Access = public)
        function self = BoundProjQpModel(varargin)
            self = self@model.QpModel(varargin{:});
        end
    end
end
