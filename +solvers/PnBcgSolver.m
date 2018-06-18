classdef PnBcgSolver < solvers.NlpSolver
    %% PnBcgSolver - Projected Newton using CG for bound constrained opt.
    % Projected Newton method using conjugate gradients to solve the
    % quadratic approximation of the objective function subject to bound
    % constraints. This method requires a projection on the constraint set.
    %
    % According to the works of Edwin A.H. Vollebregt,
    % The Bound-Constrained Conjugate Gradient Method for Non-negative
    % Matrices
    
    
    properties (SetAccess = private, Hidden = false)
        suffDec; % Sufficient decrease coefficient in line search
        maxIterLS; % Maximal number of iterations in the line search
        fid;
        cgIter;
    end
    
    properties (Hidden = true, Constant)
        LOG_HEADER = {'Iteration', 'CG iter', 'FunEvals', 'fObj', ...
            '||Pg||'};
        LOG_FORMAT = '%10s %10s %10s %15s %15s \n';
        LOG_BODY = '%10d %10d %10d %15.5e %15.5e\n';
    end % constant properties
    
    
    methods (Access = public)
        
        function self = PnBcgSolver(nlp, varargin)
            %% Constructor
            if ~ismethod(nlp, 'project')
                error('nlp doesn''t contain a project method');
            end
            
            % Gathering optional arguments and setting default values
            p = inputParser;
            p.KeepUnmatched = true;
            p.PartialMatching = false;
            p.addParameter('suffDec', 1e-4);
            p.addParameter('maxIterLS', 50); % Max iters for line search
            p.addParameter('fid', 1);
            
            p.parse(varargin{:});
            
            self = self@solvers.NlpSolver(nlp, p.Unmatched);
            
            self.suffDec = p.Results.suffDec;
            self.maxIterLS = p.Results.maxIterLS;
            self.fid = p.Results.fid;
            
            import utils.PrintInfo;
            import linesearch.armijo;
        end % constructor
        
        function self = solve(self)
            %% Solve using the Projected Newton for bounds algorithm
            
            self.solveTime = tic;
            self.iter = 1;
            self.iStop = self.EXIT_NONE;
            self.cgIter = 0;
            self.nlp.resetCounters();
            
            printObj = utils.PrintInfo('PnBcg');
            
            % Output Log
            if self.verbose >= 2
                extra = containers.Map({'suffDec', 'maxIterLS'}, ...
                    {self.suffDec, self.maxIterLS});
                printObj.header(self, extra);
                self.printf(self.LOG_FORMAT, self.LOG_HEADER{:});
            end
            
            % Project x0 to make sure it is a feasible point
            x = self.project(self.nlp.x0);
            
            % Getting obj. func, gradient and hessian at x
            [f, g, H] = self.nlp.obj(x);
            
            fOld = Inf;
            
            % Relative stopping tolerance
            self.gNorm0 = norm(g);
            rOptTol = self.rOptTol * norm(g);
            rFeasTol = self.rFeasTol * abs(f);
            
            %% Main loop
            while ~self.iStop % self.iStop == 0
                
                % Stopping criteria is the norm of the 'working' gradient
                pgnrm = norm(self.project(x - g) - x);
                
                % Output log
                self.nObjFunc = self.nlp.ncalls_fobj + ...
                    self.nlp.ncalls_fcon;
                if self.verbose >= 2
                    self.printf(self.LOG_BODY, self.iter, self.cgIter, ...
                        self.nObjFunc, f, pgnrm);
                end
                
                % Checking various stopping conditions, exit if true
                if pgnrm < rOptTol + self.aOptTol
                    self.iStop = self.EXIT_OPT_TOL;
                elseif abs(f - fOld) < rFeasTol + self.aFeasTol
                    self.iStop = self.EXIT_FEAS_TOL;
                elseif self.nObjFunc >= self.maxEval
                    self.iStop = self.EXIT_MAX_EVAL;
                elseif self.iter >= self.maxIter
                    self.iStop = self.EXIT_MAX_ITER;
                elseif toc(self.solveTime) >= self.maxRT;
                    self.iStop = self.EXIT_MAX_RT;
                end
                
                if self.iStop % self.iStop ~= 0
                    break
                end
                
                % Compute descent direction
                [xs, failed] = self.BCCG(-g, H);
                if failed
                    self.iStop = self.EXIT_INNER_FAIL;
                    break;
                end
                d = xs - x;
                
                % Compute Armijo line search
                [x, ~, failed] = linesearch.armijo(self, x, f, g, d);
                if failed
                    self.iStop = self.EXIT_MAX_ITER_LS;
                    break;
                end
                
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
            % Set solved attribute
            self.isSolved();
            
            printObj.footer(self);
        end % solve
        
        function printf(self, varargin)
            %% Printf - prints variables arguments to a file
            fprintf(self.fid, varargin{:});
        end
        
        
    end % public methods
    
    
    methods (Access = private)
        
        function [x, failed] = BCCG(self, b, A)
            %% BCCG routine
            
            % Failure flag
            failed = false;
            % Projection of current tildex stored in x
            x = self.project(zeros(self.nlp.n, 1));
            % Initialize the gradient
            g = A * x - b;
            % Initialize the binding set
            fixed = (x == self.nlp.bU & g < 0) | ...
                (x == self.nlp.bL & g > 0);
            
            % Stopping tolerance on ||r||
            rTol = norm(g) * self.aOptTol + self.aOptTol;
            for iter = 1:self.nlp.n
                
                % Free variables residual is -g, fixed variables set to 0
                r = -g;
                r(fixed) = 0;
                % Updating the direction
                if iter == 1 || ~any(fixed ~= oldFixed)
                    p = r;
                else
                    p = r + ((r' * r) / rtr) * p;
                end
                
                % Saving r' * r of the current iteration
                rtr = r' * r;
                % Compute alph
                q = A * p;
                alph = rtr / (p' * q); % p' * q := curvature condition
                
                % Update x and evaluate its projection
                tildex = x + alph * p;
                x = self.project(tildex);
                
                if any(tildex > self.nlp.bU) || any(tildex < self.nlp.bL)
                    g = A * x - b;
                else % all(xp == x)
                    g = g + alph * q;
                end
                
                % Saving the fixed variables from the current iteration
                oldFixed = fixed;
                % Updating the binding set
                fixed = (x == self.nlp.bU & g < 0) | ...
                    (x == self.nlp.bL & g > 0);
                
                % Exit if the residual convergence test is satisfied.
                if sqrt(rtr) <=  rTol || all(fixed)
                    self.cgIter = iter;
                    return
                end
                
            end % for loop
            self.cgIter = iter;
            failed = true;
        end % bccg
        
        function z = project(self, x)
            %% Project
            % Project on the bounds assuming x is full-sized.
            z = min(max(x, self.nlp.bL), self.nlp.bU);
        end
        
        function z = projectSel(self, x, ind)
            %% ProjectSel
            % Project on the bounds for a selected set of indices, assuming
            % x is of reduced size.
            z = min(max(x, self.nlp.bL(ind)), self.nlp.bU(ind));
        end
         
    end
    
end