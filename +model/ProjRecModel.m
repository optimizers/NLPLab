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
        
        function self = ProjRecModel(crit, prec, sino, geos, ...
                projSolver, varargin)
            %% Constructor
            % Inputs:
            %   - crit: Critere object from the Poly-LION repository
            %   - prec: Precond object from the Poly-Lion repository
            %   - sino: Sinogramme object from the Poly-Lion repository
            %   - geos: Geometrie_Series object from the Poly-Lion repo
            %   - projSolver: solver that contains the model to solve and a
            %   solve function.
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
            
            % Verifying the projModel type
            if ~isa(projSolver, 'solvers.NlpSolver')
               error('projSolver must be a subclass of solvers.NlpSolver');
            end
            self.projSolver = projSolver;
        end
        
        function z = project(self, x)
            %% Project - Call Solve from projSolver
            % This will call projSolver's solve function that solves the
            % projection sub-problem defined by the nlp model is possesses.
            % Input:
            %   - x: vector to project on C*x >= 0
            % Output:
            %   - xProj: the projection of x on C*x >= 0
            
            % Variable that we desire to project on the constraint set
            % Here x is \bar{x}
            self.projSolver.nlp.setPointToProject(x);
            
            % Calling the solver to solve the problem
            self.projSolver.solve();
            
            % Finding the primal variable from the dual variable
            z = self.projSolver.nlp.dualToPrimal(self.projSolver.x);
        end
        
    end
    
end
