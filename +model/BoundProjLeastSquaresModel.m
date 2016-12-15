classdef BoundProjLeastSquaresModel < model.LeastSquaresModel & model.BoundProj
    methods (Access = public)
        function self = BoundProjLeastSquaresModel(varargin)
            self = self@model.LeastSquaresModel(varargin{:});
        end
    end
end
