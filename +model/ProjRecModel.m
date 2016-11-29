classdef ProjRecModel < model.RecModel
    %% ProjRecModel - Extension of RecModel for solvers w. proj. sub.-prob.
    %   This model provides a 
    %           [z, solver] = project(self, x)
    %   method that projects on the constraint set C*x >= 0. This model is
    %   required if a projection method is used on RecModel. For instance,
    %   minConf-nlp and Cflash require a ProjRecModel.
    %
    %   This class must receive a projection model (projModel) and a 
    %   projection sub-problem solver accordingly.
    %
    %   For additional information, look in the RecModel class.
    
    properties (SetAccess = private, Hidden = false)
        % The model representing the projection problem
        projModel;
        % The solver used to solve projModel
        projSolver;
        % Proj solver's parameters
        projOptions;
    end
    
    methods (Access = public)
        function self = ProjRecModel(crit, prec, sino, geos, projModel, ...
                projSolver, projOptions, varargin)
            %% Constructor
            % Inputs:
            %   - crit: Critere object from the Poly-LION repository
            %   - prec: Precond object from the Poly-Lion repository
            %   - sino: Sinogramme object from the Poly-Lion repository
            %   - geos: Geometrie_Series object from the Poly-Lion repo
            %   - projModel: model representing the dual of the projection
            %   problem. It is used in the project function. NOTE: 
            %   generally speaking, it is way easier to solve the dual of 
            %   the projection problem than the primal.
            %   - projSolver: solver that can receive and solve projModel.
            %   - projOptions: struct containing projSolver's parameters.
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
            self.projOptions = projOptions;
            
            % Verifying the projModel type
            if ~isa(projModel, 'model.ProjModel')
                error('projModel should be a subclass of model.ProjModel');
            elseif ~isa(projSolver, 'solvers.NLPSolver')
               error('projSolver must be a subclass of solvers.NLPSolver');
            end
            self.projModel = projModel;
            self.projSolver = projSolver;
        end
        
        function [z, solver] = project(self, x)
            %% Project - Call Solve from projModel
            % This will call projModel's Solve function that solves the
            % projection sub-problem defined in the ProjModel class.
            % Input:
            %   - x: vector to project on C*x >= 0
            % Output:
            %   - xProj: the projection of x on C*x >= 0
            
            % Variable that we desire to project on the constraint set
            % Here x is \bar{x}
            self.projModel.xbar = x;
            
            solver = self.projSolver(self.projModel, self.projOptions);
            solver = solver.solve();
            
            % Finding the primal variable from the dual variable
            z = self.projModel.dualToPrimal(zProj);
        end
    end
end
