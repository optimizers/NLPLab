classdef BbSolver < solvers.NlpSolver
    %% BbSolver - Barzilai-Borwein projected gradient descent (SPG)
    % Based on the Non-monotone spectral projected gradient methods on
    % convex sets by Birgin, Martinez and Raydan
    
    properties (SetAccess = private, Hidden = false)
        % Projection calls counter
        nProj;
        memory;
        maxIterLS;
        fid;
        storedObjFunc;
        suffDec;
    end % private properties
    
    properties (Hidden = true, Constant)
        LOG_HEADER = { ...
            'Iteration', 'FunEvals', 'Projections', 'Step Length', ...
            'Function Val', '||Pg||'};
        LOG_FORMAT = '%10s %10s %10s %15s %15s %15s\n';
        LOG_BODY = '%10d %10d %10d %15.5e %15.5e %15.5e\n';
        EXIT_MSG = { ...
            'First-Order Optimality Conditions Below optTol\n', ...     % 1
            'Function value changing by less than feasTol\n', ...       % 2
            'Function Evaluations exceeds maxEval\n', ...               % 3
            'Maximum number of iterations reached\n', ...               % 4
            'Maximum number of iterations in line search reached\n', ...% 5
            };
        
        ALPH_MIN = 1e-3;
        ALPH_MAX = 1e3;
        SIG_1 = 0.1;
        SIG_2 = 0.9;
    end % constant properties
    
    
    methods (Access = public)
        
        function self = BbSolver(nlp, varargin)
            %% Constructor
            % Inputs:
            %   - nlp: a subclass of a nlp model containing the 'obj'
            %   function that returns variable output arguments among the
            %   following: objective function, gradient and hessian at x.
            %   The hessian can be a Spot operator if it is too expensive
            %   to compute. The method also supports a L-BFGS approximation
            %   of the hessian.
            %   - varargin (optional): the various parameters of the
            %   algorithm
            
            if ~ismethod(nlp, 'project')
                error('nlp doesn''t contain a project method');
            end
            
            % Gathering optional arguments and setting default values
            p = inputParser;
            p.PartialMatching = false;
            p.KeepUnmatched = true;
            p.addParameter('memory', 10);
            p.addParameter('fid', 1);
            p.addParameter('suffDec', 1e-4);
            p.addParameter('maxIterLS', 50); % Max iters for linesearch
            
            p.parse(varargin{:});
            
            self = self@solvers.NlpSolver(nlp, p.Unmatched);
            
            self.memory = p.Results.memory;
            self.maxIterLS = p.Results.maxIterLS;
            self.suffDec = p.Results.suffDec;
            self.fid = p.Results.fid;
            
            % Initialize non-monotone line search objective function array
            self.storedObjFunc = -inf(self.memory, 1);
            
            import utils.PrintInfo;
            import linesearch.nmSpectralArmijo;
        end % constructor
        
        function self = solve(self)
            %% Solve
            
            self.solveTime = tic;
            
            printObj = utils.PrintInfo('Bb');
            
            % Output Log
            if self.verbose >= 2
                % Printing header
                extra = containers.Map( ...
                    {'suffDec', 'maxIterLS', 'memory'}, ...
                    {self.suffDec, self.maxIterLS, self.memory});
                printObj.header(self, extra);
                self.printf(self.LOG_FORMAT, self.LOG_HEADER{:});
            end
            
            % Exit flag set to 0, will exit if not 0
            self.iStop = 0;
            % Resetting the counters
            self.nProj = 0;
            self.iter = 1;
            
            % Make sure point is feasible
            x = self.project(self.nlp.x0);
            % Evaluate initial point & derivative
            [f, g] = self.nlp.obj(x);
            
            % Relative stopping tolerance
            self.rOptTol = self.aOptTol * norm(g);
            self.rFeasTol = self.aFeasTol * abs(f);
            
            % Initial descent direction is the steepest descent
            alph = 1;
            
            %% Main loop
            while self.iStop == 0
                
                % Descent direction
                d = self.project(x - alph * g) - x;
                
                % Check function progression
                fOld = f;
                % Storing older values to compute BB step length
                xOld = x;
                gOld = g;
                
                % Perform a non-monotone Armijo line search
                [x, f, failed, t] = linesearch.nmSpectralArmijo(self, ...
                    x, f, g, d);
                if failed
                    self.iStop = 5;
                    break;
                end
                
                % Evaluate gradient at new x
                g = self.nlp.gobj(x);
                
                % Compute new step length according to BB rule
                alph = self.bbStepLength(xOld, x, gOld, g);
                
                % Output log
                pgnrm = norm(self.project(x - g) - x);
                if self.verbose >= 2
                    self.nObjFunc = self.nlp.ncalls_fobj + ...
                        self.nlp.ncalls_fcon;
                    self.printf(self.LOG_BODY, self.iter, ...
                        self.nObjFunc, self.nProj, t, f, pgnrm);
                end
                
                % Checking stopping conditions
                if pgnrm < self.rOptTol + self.aOptTol
                    self.iStop = 1;
                elseif abs(f - fOld) < self.rFeasTol + self.aFeasTol
                    self.iStop = 2;
                elseif self.nObjFunc > self.maxEval
                    self.iStop = 3;
                elseif self.iter >= self.maxIter
                    self.iStop = 4;
                end
                
                if self.iStop ~= 0
                    break;
                end
                
                self.iter = self.iter + 1;
            end % main loop
            
            self.x = x;
            self.fx = f;
            self.pgNorm = pgnrm;
            
            self.nObjFunc = self.nlp.ncalls_fobj + self.nlp.ncalls_fcon;
            self.nGrad = self.nlp.ncalls_gobj + self.nlp.ncalls_gcon;
            self.nHess = self.nlp.ncalls_hvp + self.nlp.ncalls_hes;
            
            %% End of solve
            self.solved = ~(self.iStop == 6 || self.iStop == 7);
            self.solveTime = toc(self.solveTime);
            
            printObj.footer(self);
        end % solve
        
        function printf(self, varargin)
            %% Printf - prints variables arguments to a file
            fprintf(self.fid, varargin{:});
        end
        
    end % public methods
    
    
    methods (Access = private)
        
        function z = project(self, x)
            %% Project - projecting x on the constraint set
            z = self.nlp.project(x);
            self.nProj = self.nProj + 1;
        end
        
        function alph = bbStepLength(self, xOld, x, gOld, g)
            %% BBStepLength - Compute Barzilai-Borwein step length
            % alph_BB = (s' * s) / (s' * y)
            s = x - xOld;
            % Denominator of Barzilai-Borwein step length
            betaBB = s' * (g - gOld);
            if betaBB < 0
                % Fall back to maximal step length
                alph = self.ALPH_MAX;
            else
                % Compute Barzilai-Borwein step length
                % y = g - gOld
                % alph_BB = (s' * s) / (s' * y)
                % Assert alph \in [alph_min, alph_max]
                alph = min(self.ALPH_MAX, ...
                    max(self.ALPH_MIN, (s' * s) / betaBB));
            end
        end % bbsteplength
        
        function alph = bbStepLength2(self, xOld, x, gOld, g)
            %% BBStepLength2 - Compute Barzilai-Borwein step length
            % % alph_BB = (s' * y) / (y' * y)
            y = g - gOld;
            % Denominator of Barzilai-Borwein step length
            betaBB = y' * y;
            if betaBB < 0
                % Fall back to maximal step length
                alph = self.ALPH_MAX;
            else
                % Compute Barzilai-Borwein step length
                % s = x - xOld
                % alph_BB = (s' * y) / (y' * y)
                % Assert alph \in [alph_min, alph_max]
                alph = min(self.ALPH_MAX, ...
                    max(self.ALPH_MIN, ((x - xOld)' * y) / betaBB));
            end
        end % bbsteplength
        
    end % private methods
    
end % class