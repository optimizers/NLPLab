
classdef TmpLiSolver < handle
    %% TmpLiSolver - Projected Newton optimization algorithm on custom set
    
    
    properties (Access = public)
        % Execution time
        solveTime;
    end
    
    properties (SetAccess = private, Hidden = false)
        % NlpModel
        nlp;
        % x upon termination
        x;
        % Objective function value at x
        fx;
        % Norm of projected gradient at x
        pgNorm;
        % Iteration counter
        iter;
        % Exit flag
        iStop;
        % Solved flag
        solved;
        % Objective function calls counter
        nObjFunc;
        nGrad;
        nHess;
        % optTol relative to the initial gradient norm
        stopTol;
        relFuncTol;
    end
    
    properties (Access = private, Hidden = false)
        verbose; % 0, 1 or 2
        % Note: optTol := || P[x - g] - x ||
        optTol; % Optimality tolerance in the main problem
        funcTol;
        maxIter; % Maximum number of iterations
        maxEval; % Maximum number of calls to objective function
        suffDec; % Sufficient decrease coefficient in line search
        maxIterLS; % Maximal number of iterations in the line search
        fid;
        eqTol;
        lsFunc; % Line search function
    end
    
    properties (Hidden = true, Constant)
        LOG_HEADER = {'Iteration', 'FunEvals', 'fObj', '||Pg||', ...
            'nWorking'};
        LOG_FORMAT = '%10s %10s %15s %15s %10s\n';
        LOG_BODY = '%10d %10d %15.5e %15.5e %10d\n';
        EXIT_MSG = { ...
            ['All variables are at their bound and no further', ...
            ' progress is possible\n'], ...                             % 1
            'All working variables satisfy optimality condition\n', ... % 2
            'Function value changing by less than funcTol\n', ...       % 3
            'Function Evaluations exceeds maxEval\n', ...               % 4
            'Maximum number of iterations reached\n', ...               % 5
            'Maximum number of iterations in line search reached\n', ...% 6
            };
    end % constant properties
    
    
    methods (Access = public)
        
        function self = TmpLiSolver(nlp, varargin)
            %% Constructor
            if ~isa(nlp, 'model.NlpModel')
                error('Model must be a NlpModel');
            elseif ~ismethod(nlp, 'project')
                error('nlp doesn''t contain a project method');
            elseif ~ismethod(nlp, 'eqProject')
                error('nlp doesn''t contain an eqProject method');
            elseif ~isprop(nlp, 'normJac') && ~ismethod(nlp, 'normJac')
                error('nlp doesn''t contain a normJac attribute');
            end
            self.nlp = nlp;
            
            % Gathering optional arguments and setting default values
            p = inputParser;
            p.KeepUnmatched = true;
            p.PartialMatching = false;
            p.addParameter('verbose', 2);
            p.addParameter('optTol', 1e-5);
            p.addParameter('eqTol', 1e-10);
            p.addParameter('maxIter', 5e2);
            p.addParameter('maxEval', 5e2);
            p.addParameter('suffDec', 1e-4);
            p.addParameter('maxIterLS', 50); % Max iters for line search
            p.addParameter('funcTol', eps);
            p.addParameter('exactLS', false);
            p.addParameter('fid', 1);
            
            p.parse(varargin{:});
            self.verbose = p.Results.verbose;
            self.optTol = p.Results.optTol;
            self.eqTol = p.Results.eqTol;
            self.maxIter = p.Results.maxIter;
            self.maxEval = p.Results.maxEval;
            self.suffDec = p.Results.suffDec;
            self.maxIterLS = p.Results.maxIterLS;
            self.funcTol = p.Results.funcTol;
            self.fid = p.Results.fid;
            
        end % constructor
        
        function self = solve(self)
            %% Solve using the Projected Newton for bounds algorithm
            self.solveTime = tic;
            self.iter = 1;
            self.iStop = 0;
            
            % Output Log
            if self.verbose >= 2
                self.printHeaderFooter('header');
                self.printf(self.LOG_FORMAT, self.LOG_HEADER{:});
            end
            
            % Project x0 to make sure it is a feasible point
            x = self.nlp.project(self.nlp.x0);
            
            % Getting obj. func, gradient and hessian at x
            [f, g, H] = self.nlp.obj(x);
            
            fOld = Inf;
            
            % Relative stopping tolerance
            self.stopTol = self.optTol * norm(g);
            self.relFuncTol = self.funcTol * abs(f);
            
            %% Main loop
            while self.iStop == 0
                
                % Get working set of variables
                fixed = self.getFixedSet(x, -g);
                
                % Stopping criteria is the norm of the 'working' gradient
                pgnrm = norm(self.nlp.project(x - g) - x);
                
                % Output log
                self.nObjFunc = self.nlp.ncalls_fobj + ...
                    self.nlp.ncalls_fcon;
                if self.verbose >= 2
                    fprintf(self.LOG_BODY, self.iter, self.nObjFunc, f, ...
                        pgnrm, sum(~fixed));
                end
                
                % Checking various stopping conditions, exit if true
                if ~any(~fixed) % all(fixed)
                    self.iStop = 1;
                elseif pgnrm < self.stopTol + self.optTol
                    self.iStop = 2;
                elseif abs(f - fOld) < self.relFuncTol + self.funcTol
                    self.iStop = 3;
                elseif self.nObjFunc >= self.maxEval
                    self.iStop = 4;
                elseif self.iter >= self.maxIter
                    self.iStop = 5;
                end
                
                if self.iStop ~= 0
                    break
                end
                
                %                 % Compute Newton direction, H*d = -g
                %                 d2 = self.newtonDir(g, H);
                %                 % Project on C(fixed, :) * d = 0
                %                 d2 = self.nlp.eqProject(d2, fixed);
                
                d = self.nlp.minEqProject(g, H, fixed);

                % Compute Armijo line search
                x = self.armijo(x, f, g, d);
                
                % Saving old f to check progression
                fOld = f;
                
                % Updating objective function, gradient and hessian
                [f, g, H] = self.nlp.obj(x);
                
                self.iter = self.iter + 1;
            end % main loop
            
            self.x = x;
            self.fx = f;
            self.pgNorm = pgnrm;
            
            self.nObjFunc = self.nlp.ncalls_fobj + self.nlp.ncalls_fcon;
            self.nGrad = self.nlp.ncalls_gobj + self.nlp.ncalls_gcon;
            self.nHess = self.nlp.ncalls_hvp + self.nlp.ncalls_hes;
            
            %% End of solve
            self.solveTime = toc(self.solveTime);
            self.solved = ~(self.iStop == 8 || self.iStop == 9);
            
            if self.verbose
                self.printf('\nEXIT TMPLI: %s\nCONVERGENCE: %d\n', ...
                    self.EXIT_MSG{self.iStop}, self.solved);
                self.printf('||Pg|| = %8.1e\n', self.pgNorm);
                self.printf('Stop tolerance = %8.1e\n', self.stopTol);
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
                    self.printf([repmat('\t', 1, 3), 'TmpLi Solver \n']);
                    self.printf('%s\n\n', ['*', repmat('-',1,58), '*']);
                    self.printf(self.nlp.formatting())
                    self.printf('\nParameters\n----------\n')
                    self.printf('%-15s: %3s %8i', 'maxIter', '', ...
                        self.maxIter);
                    self.printf('\t%-15s: %3s %8.1e\n', ' optTol', '', ...
                        self.optTol);
                    self.printf('%-15s: %3s %8.1e', 'suffDec', '', ...
                        self.suffDec);
                    self.printf('\t%-15s: %3s %8d\n', ' maxIterLS', '', ...
                        self.maxIterLS);
                    self.printf('\n');
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
        
        function fixed = getFixedSet(self, x, d)
            %% GetFixedSet - Finds the fixed set for proj. Newton step
            % Inputs:
            %   - x: current point
            %   - d: descent direction
            % Outputs:
            %   - fixed: bool array of fixed variables
            
            Cx = self.nlp.fcon(x); % C*x
            Cd = self.nlp.fcon(d); % C*d
            
            % Smallest approximation is eps
            appZero = max(self.eqTol * self.nlp.normJac * norm(x), eps);
            
            % Find gradient fixed set
            fixed = (Cx - self.nlp.cL < appZero & Cd < 0) | ...
                (self.nlp.cU - Cx < appZero & Cd > 0);
        end
        
        function d = newtonDir(self, g, H)
            %% NewtonDir - computes a Newton descent direction
            % Solves the equation H * d = -g, using the gradient and
            % hessian provided as input arguments, assuming they are of
            % reduced size. This descent direction should only be computed
            % on the free variables.
            
            % Different methods could be used. Using PCG for now.
            [d, ~] = pcg(H, -g, max(1e-5 * self.optTol, 1e-12), 1e5);
        end
        
        function xNew = armijo(self, x, f, g, d)
            %% Armijo - Armijo line search
            % Perform an Armijo line search on the reduced
            % variables according to 'working'. This function assumes that
            % freeX, freeG and freeH are already reduced. However, xNew
            % must be full-sized since calls will be made to the objective
            % function.
            iterLS = 1;
            t = 1;
            while true
                % Recompute trial step on free variables
                xNew = self.nlp.project(x + t * d);
                % Update objective function value
                fNew = self.nlp.obj(xNew);
                % Checking exit conditions
                if (f - fNew) >= (self.suffDec * t * g' * d)
                    % Armijo condition satisfied
                    return;
                elseif iterLS >= self.maxIterLS
                    % Maximal number of iterations reached, abort
                    self.iStop = 6;
                    return;
                end
                % Decrease step size
                t = t / 2;
                iterLS = iterLS + 1;
            end
        end
        
        function x = backtracking(self, x, f, g, d)
            %% Backtracking Line Search
            % Applies an interpolation procedure (half step) if the new
            % value doesn't improve the value of the objective function.
            
            iterLS = 1;
            t = 1;
            % Check if decrease is sufficient
            while true
                
                % Evaluate new point
                xNew = self.nlp.project(x + t * d);
                % Evaluate objective function at new point
                fNew = self.nlp.obj(xNew);
                
                if fNew <= f + self.suffDec * g' * (xNew - x)
                    return;
                elseif iterLS >= self.maxIterLS
                    % Maximal number of iterations reached, abort
                    self.iStop = 6;
                    return;
                end
                
                t = 0.5 * t;
                iterLS = iterLS + 1;
            end
        end % backtracking
        
    end
    
end