classdef PnbSolver < solvers.NlpSolver
    %% PnbSolver - Projected Newton algorithm for bounded optimization
    
    properties (SetAccess = private, Hidden = false)
        suffDec; % Sufficient decrease coefficient in line search
        maxIterLS; % Maximal number of iterations in the line search
        fid;
        lsFunc; % Line search function
        exactLS;
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
        
        function self = PnbSolver(nlp, varargin)
            %% Constructor
            
            % Gathering optional arguments and setting default values
            p = inputParser;
            p.KeepUnmatched = true;
            p.PartialMatching = false;
            p.addParameter('suffDec', 1e-4);
            p.addParameter('maxIterLS', 50); % Max iters for line search
            p.addParameter('exactLS', false);
            p.addParameter('fid', 1);
            
            p.parse(varargin{:});
            
            self = self@solvers.NlpSolver(nlp, p.Unmatched);
            
            self.suffDec = p.Results.suffDec;
            self.maxIterLS = p.Results.maxIterLS;
            self.exactLS = p.Results.exactLS;
            self.fid = p.Results.fid;
            
            % Exact line search is only implemented for quadratic or least
            % squares models
            if self.exactLS && ...
                    (isa(self.nlp, 'model.LeastSquaresModel') || ...
                    isa(self.nlp, 'model.QpModel'))
                self.lsFunc = @(xNew, f, x, g, d, H, working) ...
                    self.restrictedExact(xNew, f, x, g, d, H, working);
            else
                % Otherwise, default to Armijo
                self.lsFunc = @(xNew, f, x, g, d, H, working) ...
                    self.restrictedArmijo(xNew, f, x, g, d, H, working);
            end
            
            import utils.PrintInfo;
%             import linesearch.redProjArmijo;
        end % constructor
        
        function self = solve(self)
            %% Solve using the Projected Newton for bounds algorithm
            
            self.solveTime = tic;
            self.iter = 1;
            self.iStop = 0;
            
            printObj = utils.PrintInfo('Pnb');
            
            % Output Log
            if self.verbose >= 2
                extra = containers.Map( ... 
                    {'suffDec', 'maxIterLS', 'exactLS'}, ...
                    {self.suffDec, self.maxIterLS, self.exactLS});
                printObj.header(self, extra);
                self.printf(self.LOG_FORMAT, self.LOG_HEADER{:});
            end
            
            % Project x0 to make sure it is a feasible point
            x = self.project(self.nlp.x0);
            
            % Getting obj. func, gradient and hessian at x
            [f, g, H] = self.nlp.obj(x);
            
            fOld = Inf;
            
            % Relative stopping tolerances
            self.rOptTol = self.aOptTol * norm(g);
            self.rFeasTol = self.aFeasTol * abs(f);
            
            %% Main loop
            while self.iStop == 0
                
                % Get working set of variables
                working = self.getWorkingSet(x, g, H);
                
                % Stopping criteria is the norm of the 'working' gradient
                pgnrm = norm(g(working));
                
                % Output log
                self.nObjFunc = self.nlp.ncalls_fobj + ...
                    self.nlp.ncalls_fcon;
                if self.verbose >= 2
                    self.printf(self.LOG_BODY, self.iter, ...
                        self.nObjFunc, f, pgnrm, sum(working));
                end
                
                % Checking various stopping conditions, exit if true
                if ~any(working)
                    self.iStop = 1;
                elseif pgnrm < self.rOptTol + self.aOptTol
                    self.iStop = 2;
                elseif abs(f - fOld) < self.rFeasTol + self.aFeasTol
                    self.iStop = 3;
                elseif self.nObjFunc >= self.maxEval
                    self.iStop = 4;
                elseif self.iter >= self.maxIter
                    self.iStop = 5;
                end
                
                if self.iStop ~= 0
                    break
                end
                
                % Saving fixed variables in xNew, x will be overwritten
                xNew = x;
                
                % We can truncate x, g and H as we are only working on the
                % free variables.
                x = x(working);
                g = g(working);
                H = H(working, working);
                
                % Compute Newton direction for free variables only
                d = self.newtonDir(g, H);
                
                % Compute restricted projected Armijo line search
                xNew = self.lsFunc(xNew, f, x, g, d, H, working);
                
                % Taking step, restore x to full-sized
                x = xNew;
                
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
            
            printObj.footer(self);
        end % solve
        
        function printf(self, varargin)
            %% Printf - prints variables arguments to a file
            fprintf(self.fid, varargin{:});
        end
        
    end % public methods
    
    
    methods (Access = private)
        
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
        
        function working = getWorkingSet(self, x, g, H)
            %% GetWorkingSet - Finds the fixed set for proj. Newton step
            % Inputs:
            %   - x: current point
            %   - g: gradient at current point
            %   - H: hessian at current point
            % Outputs:
            %   - working: bool array of free variables
            
            % Find gradient fixed set
            gFixed = (x == self.nlp.bL & g > 0) | ...
                (x == self.nlp.bU & g < 0);
            
            % Save gradient fixed set
            fixed = gFixed;
            
            % However, we will later compute a Newton direction instead of
            % a steepest descent direction. Therefore, computing a Newton
            % direction on the remaining free variables can help us
            % identify more variables that should be fixed.
            
            % Compute a Newton direction from reduced g & H
            d = self.newtonDir(g(~gFixed), H(~gFixed, ~gFixed));
            
            % We restrict x to the free variables
            x = x(~gFixed);
            
            % Update the gradient fixed set with the Newton fixed set
            % fixed := gradient fixed set | Newton fixed set
            fixed(~gFixed) = (x == self.nlp.bL(~gFixed) & d < 0) | ...
                (x == self.nlp.bU(~gFixed) & d > 0);
            
            % Finally, the working set represents the free variables
            working = ~fixed;
        end
        
        function d = newtonDir(self, g, H)
            %% NewtonDir - computes a Newton descent direction
            % Solves the equation H * d = -g, using the gradient and
            % hessian provided as input arguments, assuming they are of
            % reduced size. This descent direction should only be computed
            % on the free variables.
            
            % Different methods could be used. Using PCG for now.
            [d, ~] = pcg(H, -g, self.aOptTol + self.rOptTol, ...
                max(1e4, self.nlp.n));
        end
        
        function xNew = restrictedArmijo(self, xNew, f, x, g, d, ~, ...
                working)
            %% RestrictedArmijo - Armijo line search on the restricted vars
            % Perform a projected Armijo line search on the reduced
            % variables according to 'working'. This function assumes that
            % freeX, freeG and freeH are already reduced. However, xNew
            % must be full-sized since calls will be made to the objective
            % function.
            iterLS = 1;
            t = 1;
            while true
                % Recompute trial step on free variables
                xNew(working) = self.projectSel(x + t * d, working);
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
        
        function xNew = restrictedExact(self, xNew, ~, x, g, d, H, working)
            %% RestrictedExact
            % Exact line search restricted to the free variables. xNew is
            % projected at the end to ensure the value remains within the
            % bounds.
            % Step length must stay between 0 and 1
            t = min(max((d' * -g) / (d' * H * d), eps), 1);
            % Take step and project to make sure bounds are satisfied
            xNew(working) = self.projectSel(x + t * d, working);
        end
        
    end
    
end