classdef CFlashSolver < solvers.NLPSolver
    %% CFlashSolver - Calls the CFlash solver
    % 
    
    properties (SetAccess = private)
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    methods (Access = public)
        
        function self = CFlashSolver(nlp, varargin)
            self = self@solvers.NLPSolver(nlp, varargin{:});
        end
        
        function self = solve(self)
            %% Solve
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
end