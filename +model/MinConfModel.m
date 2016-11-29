classdef MinConfModel < model.RecModel
    %% MinConfModel - Custom model applying minConf to reconstruction prob.
    %   The Solve function calls minConf_PQN on the functions specifically
    %   defined in this class to represent the tomographic
    %   reconstruction problem. The projection function required by
    %   minConf_PQN is obtained by solving another projection
    %   problem, which in turn is solved by minConf_TMP.
    %
    %   This class was developped for the following problem
    %   min_x 1/2 || PCx - y ||^2 + \lambda\phi(Cx)
    %   st      Cx >= 0
    %   where P and C are matrices that don't have an explicit form, i.e.
    %   only their matrix-vector product is available.
    %
    %   This model is only compatible with the structure of the tomographic
    %   reconstruction algorithm made by Poly-LION. The terms in the
    %   objective function should be provided by a Critere object, whereas
    %   the preconditionner should be a Precond object. This model also
    %   expects the sinogram to be a Sinogramme object and the geos a
    %   GeometrieSeries object.
    %   ---
    
    properties (SetAccess = private, Hidden = false)
        % The model representing the projection problem
        projModel;
        % minConf parameters
        options;
        % Internal parameters
        xSol;
    end
    
    methods (Access = public)
        function self = MinConfModel(crit, prec, sino, geos, projModel, ...
                options, varargin)
            %% Constructor
            % Inputs:
            %   - crit: Critere object from the Poly-LION repository
            %   - prec: Precond object from the Poly-Lion repository
            %   - sino: Sinogramme object from the Poly-Lion repository
            %   - geos: Geometrie_Series object from the Poly-Lion repo
            %   - projModel: model representing the dual of the projection
            %   problem. It is used in the project & gradProject functions.
            %   NOTE: generally speaking, it is way easier to solve the
            %   dual of the projection problem than the primal.
            %   - options: struct containing Cflash's parameters
            %   - mu0: (optional) initial vector for the reconstruction
            %   problem
            %   - name: (optional) name of the nlp model
            
            % Gathering optinal arguments and setting default values
            p = inputParser;
            p.addParameter('mu0', []);
            p.addParameter('name', '');
            % Parsing input arguments
            p.parse(varargin{:});
            % Setting values
            mu0 = p.Results.mu0;
            name = p.Results.name;
            
            % Calling the RecModel superclass (inputs will be checked here)
            self = self@model.RecModel(crit, prec, sino, geos, mu0, name);
            
            % Storing the options
            self.options = options;
            
            % Verifying the projModel type
            if ~isa(projModel, 'model.ProjModel')
                error('projModel should be a subclass of model.ProjModel');
            end
            self.projModel = projModel;
        end
        
        function [xSol, tSol, solver] = Solve(self)
            %% Solve the reconstruction problem using MinConf_PQN
            % Call PQN solver and solve the problem
            tic;
            solver = MinConf_PQN(self, self.options);
            tSol = toc;
            xSol = solver.x;
            self.xSol = xSol;
        end
        
        function [z, solver] = project(self, x)
            %% Project - Call Solve from projModel
            % This will call projModel's Solve function that solves the
            % projection sub-problem defined in the ProjModel class.
            [z, solver] = self.projModel.Solve(x);
        end
    end
end
