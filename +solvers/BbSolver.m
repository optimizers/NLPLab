classdef BbSolver < solvers.NlpSolver
    %% BbSolver - Barzilai-Borwein projected gradient descent (SPG)
    % Based on the Non-monotone spectral projected gradient methods on
    % convex sets by Birgin, Martinez and Raydan
    
    properties (SetAccess = private, Hidden = false)
        % Norm of projected gradient at x
        pgNorm;
        % Exit flag
        iStop;
        % Projection calls counter
        nProj;
    end % gettable private properties
    
    properties (Access = private, Hidden = false)
        % Internal parameters
        verbose; % 0, 1 or 2
        maxEval;
        memory;
        maxIterLS;
        fid;
        storedObjFunc;
    end % private properties
    
    properties (Hidden = true, Constant)
        LOG_HEADER = { ...
            'Iteration', 'FunEvals', 'Projections', 'Step Length', ...
            'Function Val', '||Pg||'};
        LOG_FORMAT = '%10s %10s %10s %15s %15s %15s\n';
        LOG_BODY = '%10d %10d %10d %15.5e %15.5e %15.5e\n';
        EXIT_MSG = { ...
            'First-Order Optimality Conditions Below optTol\n', ...     % 2
            'Function value changing by less than frTol\n', ...         % 4
            'Function Evaluations exceeds maxEval\n', ...               % 5
            'Maximum number of iterations reached\n', ...               % 6
            'Maximum number of iterations in line search reached\n', ...% 7
            };
        
        ALPH_MIN = 1e-3;
        ALPH_MAX = 1e3;
        SUFF_DEC = 1e-4;
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
            p.addParameter('verbose', 2);
            p.addParameter('maxEval', 5e2);
            p.addParameter('memory', 10);
            p.addParameter('fid', 1);
            p.addParameter('maxIterLS', 50); % Max iters for linesearch
            
            p.parse(varargin{:});
            
            self = self@solvers.NlpSolver(nlp, p.Unmatched);
            
            self.verbose = p.Results.verbose;
            self.maxEval = p.Results.maxEval;
            self.memory = p.Results.memory;
            self.maxIterLS = p.Results.maxIterLS;
            self.fid = p.Results.fid;
            
            % Initialize non-monotone line search objective function array
            self.storedObjFunc = -inf(self.memory, 1);
        end % constructor
        
        function self = solve(self)
            %% Solve
            
            self.solveTime = tic;
            
            % Output Log
            if self.verbose >= 2
                % Printing header
                self.printHeaderFooter('header');
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
                [x, f, t] = self.nmArmijo(x, f, g, d);
                
                % Evaluate gradient at new x
                g = self.nlp.gobj(x);
                
                % Output log
                pgnrm = norm(self.project(x - g) - x);
                if self.verbose >= 2
                    self.nObjFunc = self.nlp.ncalls_fobj + ...
                        self.nlp.ncalls_fcon;
                    fprintf(self.LOG_BODY, self.iter, ...
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
                
                % Compute new step length according to BB rule
                alph = self.bbStepLength(xOld, x, gOld, g);
                
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
            if self.verbose
                self.printf('\nEXIT Bb: %s\nCONVERGENCE: %d\n', ...
                    self.EXIT_MSG{self.iStop}, self.solved);
                self.printf('||Pg|| = %8.1e\n', self.pgNorm);
                self.printf('Stop tolerance = %8.1e\n', self.rOptTol);
            end
            if self.verbose >= 2
                self.printHeaderFooter('footer');
            end
        end % solve
        
    end % public methods
    
    
    methods (Access = private)
        
        function printHeaderFooter(self, msg)
            switch msg
                case 'header'
                    % Print header
                    self.printf('\n');
                    self.printf('%s\n', ['*', repmat('-',1,58), '*']);
                    self.printf([repmat('\t', 1, 3), 'BbSolver \n']);
                    self.printf('%s\n\n', ['*', repmat('-',1,58), '*']);
                    self.printf(self.nlp.formatting())
                    self.printf('\nParameters\n----------\n')
                    self.printf('%-15s: %3s %8d', 'maxIter', '', ...
                        self.maxIter);
                    self.printf('\t%-15s: %3s %8d\n', ' maxEval', '', ...
                        self.maxEval);
                    self.printf('%-15s: %3s %8.1e', 'optTol', '', ...
                        self.aOptTol);
                    self.printf('%-15s: %3s %8.1e', 'suffDec', '', ...
                        self.suffDec);
                    self.printf('\t%-15s: %3s %8d\n', ' memory', '', ...
                        self.memory);
                case 'footer'
                    % Print footer
                    self.printf('\n')
                    self.printf(' %-27s  %6i     %-17s  %15.8e\n', ...
                        'No. of iterations', self.iter, ...
                        'Objective value', self.fx);
                    t1 = self.nlp.ncalls_fobj + self.nlp.ncalls_fcon;
                    t2 = self.nlp.ncalls_gobj + self.nlp.ncalls_gcon;
                    self.printf(' %-27s  %6i     %-17s    %6i\n', ...
                        'No. of calls to objective' , t1, ...
                        'No. of calls to gradient', t2);
                    self.printf(' %-27s  %6i \n', ...
                        'No. of Hessian-vector prods', ...
                        self.nlp.ncalls_hvp + self.nlp.ncalls_hes);
                    self.printf('\n');
                    tt = self.solveTime;
                    t1 = self.nlp.time_fobj + self.nlp.time_fcon;
                    t1t = round(100 * t1/tt);
                    t2 = self.nlp.time_gobj + self.nlp.time_gcon;
                    t2t = round(100 * t2/tt);
                    self.printf([' %-24s %6.2f (%3d%%)  %-20s %6.2f', ...
                        '(%3d%%)\n'], 'Time: function evals' , t1, t1t, ...
                        'gradient evals', t2, t2t);
                    t1 = self.nlp.time_hvp + self.nlp.time_hes;
                    t1t = round(100 * t1/tt);
                    self.printf([' %-24s %6.2f (%3d%%)  %-20s %6.2f', ...
                        '(%3d%%)\n'], 'Time: Hessian-vec prods', t1, ...
                        t1t, 'total solve', tt, 100);
                otherwise
                    error('Unrecognized case in printHeaderFooter');
            end % switch
        end % printHeaderFooter
        
        function printf(self, varargin)
            %% Printf - prints variables arguments to a file
            fprintf(self.fid, varargin{:});
        end
        
        function z = project(self, x)
            %% Project - projecting x on the constraint set
            z = self.nlp.project(x);
            self.nProj = self.nProj + 1;
        end
        
        function alph = bbStepLength(self, xOld, x, gOld, g)
            %% BBStepLength - Compute Barzilai-Borwein step length
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
        
        function [xNew, fNew, t] = nmArmijo(self, x, f, g, d)
            %% NmArmijo - Non-monotone Armijo Line Search
            
            % Update stored objective function values
            self.storedObjFunc(mod(self.iter, self.memory) + 1) = f;
            % Redefine f as the maximum
            fMax = max(self.storedObjFunc);
            
            iterLS = 1;
            t = 1;
            delta = g' * d;
            while true
                
                xNew = x + t * d;
                fNew = self.nlp.obj(xNew);
                
                if fNew <= fMax + self.SUFF_DEC * t * delta
                    % Armijo condition met
                    return
                elseif iterLS >= self.maxIterLS
                    % Maximal number of iterations reached, abort
                    self.iStop = 7;
                    return;
                end
                
                % Compute new trial step length
                tTemp = (-t^2 * delta) / (2 * ( fNew - f - t * delta));
                % Check if new step length is valid
                if tTemp >= self.SIG_1 && tTemp <= self.SIG_2
                    t = tTemp;
                else
                    t = t / 2;
                end
                
                iterLS = iterLS + 1;
            end
        end % nmarmijo
        
    end % private methods
    
end % class