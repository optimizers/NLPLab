classdef BCFlashSolver < solvers.NLPSolver
    %% BCFlashSolver - Calls the BCFlash solver
    % 
    
    properties (SetAccess = private)
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    methods (Access = public)
        
        function self = BCFlashSolver(nlp, varargin)
            self = self@solvers.NLPSolver(nlp, varargin{:});
        end
        
        function self = solve(self)
            %% Solve
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
end