classdef ProjRecModel < model.RecModel
    %% ProjRecModel - Extension of RecModel for solvers with projections
    %   This model provides a 
    %           z = project(self, x)
    %   method that projects on the constraint set C*x >= 0. This model is
    %   required if a RecModel is solved using projections on the
    %   constraint set.
    %
    %   This class must receive a projection solver which's "solve"
    %   function comes down to projecting x on C*x >= 0.
    %
    %   For additional information, look in the RecModel and ProjModel
    %   classes.

    
    properties (SetAccess = private, Hidden = false)
        % The solver used to solve projModel
        projSolver;
        solved;
        normJac;
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
            
            % Storing the projection sub-problem solver
            self.projSolver = projSolver;
            % Getting projModel's normJac property
            self.normJac = self.projSolver.nlp.normJac;
        end
        
        function z = project(self, x)
            %% Project - Call Solve from projSolver
            % This will call projSolver's solve function that solves the
            % projection sub-problem. Note that we solve the dual
            % projection problem, which comes down to a bounded least
            % squares problem, and is much easier to solve. Look inside
            % ProjModel for more information.
            % Input:
            %   - x: vector to project on C*x >= 0
            % Output:
            %   - xProj: the projection of x on C*x >= 0
            
            % Variable that we desire to project on the constraint set
            % Here x is \bar{x}
            self.projSolver.nlp.setPointToProject(x);
            
            % Calling the solver to solve the problem
            self.projSolver.solve();
            
            self.solved = self.projSolver.solved;
            
            % Finding the primal variable from the dual variable
            z = self.projSolver.nlp.dualToPrimal(self.projSolver.x);
        end
        
        function z = projectMixed(self, x, fixed)
            %% ProjectMixed
            % This function computes the projection of x on C*x >= 0 while
            % staying on the active face at x, defined by the logical array
            % fixed. In other words, we search x such that
            %
            %   min     1/2 || x - \bar{x} ||^2
            %     x     A*C*x >= 0,
            %           B*C*x = 0,
            %
            % where B is the restriction matrix defined by "fixed" and A
            % contains the other rows of the identity. This problem is 
            % highly similar to the original projection sub-problem, thus 
            % the same solver and model can be reused.
            % Input:
            %   - x: vector to project : {x | A*C*x >= 0, B*C*x = 0}
            % Output:
            %   - xProj: the projection of x
            
            % Variable that we desire to project on the constraint set
            % Here x is \bar{x}
            self.projSolver.nlp.setPointToProject(x);
            
            % Temporarily set the components B*z unbounded, where z are the
            % Lagrange multipliers of C*x >= 0.
            bL = self.projSolver.nlp.bL(fixed);
            jLow = self.projSolver.nlp.jLow(fixed);
            self.projSolver.nlp.bL(fixed) = -inf;
            self.projSolver.nlp.jLow(fixed) = false;
            
            % Calling the solver to solve the problem
            self.projSolver.solve();
            
            % Reset the bounds to what they were
            self.projSolver.nlp.bL(fixed) = bL;
            self.projSolver.nlp.jLow(fixed) = jLow;
            
            self.solved = self.projSolver.solved;
            
            % Finding the primal variable from the dual variable
            z = self.projSolver.nlp.dualToPrimal(self.projSolver.x);
        end
        
        % function z = projectActiveFace(self, x)
        % ... is not implemented here. See CflashSolver for a generic 
        % implementation.
    end
    
end
