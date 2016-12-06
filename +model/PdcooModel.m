classdef PdcooModel < model.RecModel
    %% PDCOOModel - Solving reconstruction problem using PDCOO solver
    %   This class was developped to solve the problem represented by
    %   RecModel using the PDCOO solver. Therefore, make sure that the 
    %   PDCOO folder is on MATLAB's path before running.
    %
    %   This model is only compatible with the structure of the tomographic
    %   reconstruction algorithm made by Poly-LION (as is the case for 
    %   RecModel). The terms in the objective function should be provided 
    %   by a Critere object, whereas the preconditionner should be a 
    %   Precond object. This model also expects the sinogram to be a 
    %   Sinogramme object and the geos a GeometrieSeries object.


    %% Properties
    properties (SetAccess = private, Hidden = false)
        xSol;
        options;
    end
    
    
    %% Public methods
    methods (Access = public)
        
        function self = PdcooModel(crit, prec, sino, geos, options, ...
                mu0, name)
            %% Constructor
            % Verifying if the name and mu0 fields exist (required for
            % NlpModel constructor in RecModel)
            if nargin < 6
                mu0 = [];
                name = '';
            elseif nargin < 7
                name = '';
            end
            
            % Calling the RecModel superclass
            self = self@model.RecModel(crit, prec, sino, geos, mu0, name);
            
            % Checking for required parameters, all other parameters have
            % default values in the PDCOO solver anyways
            if ~isfield(self.options, 'formulation')
                self.options.formulation = 'K2';
            end
            if ~isfield(self.options, 'solver')
                self.options.solver = 'MINRES';
            end
            if ~isfield(self.options, 'innerPrint')
                self.options.innerPrint = 0;
            end
            if ~isfield(self.options, 'backtrack')
                self.options.backtrack = true;
            end
            
            % Storing the options
            self.options = options;
        end
        
        function [xSol, tSol, solver] = Solve(self)
            %% Calling the PDCOO solver to solve the convex problem
            %   Inputs:
            %   - self: the convex model representing the problem to solve
            %   with the PDCOO algorithm
            %   Output:
            %   - xSol: the solution obtained with the custom PDCOO
            %   algorithm
            
            import model.slackmodel_spot;
            % Building custom class from previous parameters
            classname = build_variant(self.options.formulation, ...
                self.options.solver);
            
            self.options.file_id = 1;
            self.options.Print = 1;
            
            if self.options.innerPrint
                fprintf(self.options.file_id, ...
                    ('\n    Name    Objectif   Presid   Dresid'), ...
                    ('Cresid   PDitns   Inner     Time      D2 * r\n\n'));
            end
            
            % Building the problem to solve
            slack = slackmodel_spot(self);
            
            % Checking another field from self.options
            if ~isfield(self.options, 'Maxiter')
                self.options.Maxiter = min(max(30, slack.n), 100);
            end
            
            % Initializing a few internal parameters
            Anorm = normest(slack.gcon(slack.x0), 1.0e-3);
            self.options.x0 = slack.x0;
            self.options.x0(slack.jLow) = slack.bL(slack.jLow) + 1;
            self.options.x0(slack.jUpp) = slack.bU(slack.jUpp) - 1;
            self.options.x0(slack.jTwo) = (slack.bL(slack.jTwo) + ...
                slack.bU(slack.jTwo)) / 2;
            self.options.xsize = max(norm(self.options.x0, inf), 1);
            self.options.zsize = max(norm(slack.gobj(slack.x0), inf) + ...
                sqrt(slack.n) * Anorm, 1);
            self.options.z0 = self.options.zsize * ones(slack.n, 1);
            self.options.y0 = zeros(slack.m, 1);
            self.options.mu0 = self.options.zsize;
            
            % Calling custom class constructor
            solver = eval([classname, '(slack, self.options)']);
            % Solving the problem -- calling the PDCOO solver
            tic;
            solver.solve();
            tSol = toc;
            % Ouput
            if self.options.innerPrint
                fprintf(solver.file_id, ...
                    ['\n%12s   %11.4e   %6.0f   %6.0f   %6.0f   %6d'], ...
                    ['%6d   %7.2f s' '%11.4e\n'], slack.name, ...
                    slack.fobj(solver.x), log10(solver.Pinf), ...
                    log10(solver.Dinf), log10(solver.Cinf0), ...
                    solver.PDitns, solver.inner_total, solver.time, ...
                    self.options.d2^2 * norm(solver.y));
            end
            xSol = solver.x;
            % xSol contains both "x" and the slack variables
            xSol = xSol(~solver.slack.islack);
            self.xSol = xSol;
        end
        
    end
    
end
