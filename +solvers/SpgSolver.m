classdef SpgSolver < solvers.NlpSolver
    %% SpgSolver - Calls the MinConf_SPG solver
    % Original documentation follows:
    % ---------------------------------------------------------------------
    % function [x,f,self.nObjFunc,self.nProj] = minConf_SPG(funObj,x,
    % funProj, options)
    %
    % Function for using Spectral Projected Gradient to solve problems of
    % the form
    %   min funObj(x) s.t. x in C
    %
    %   @funObj(x): function to minimize (returns gradient as second
    %               argument)
    %   @funProj(x): function that returns projection of x onto C
    %
    %   options:
    %       verbose: level of verbosity (0: no output, 1: final,
    %                     2: iter (default), 3: debug)
    %       optTol: tolerance used to check for optimality (default: 1e-5)
    %       progTol: tolerance used to check for lack of progress (default:
    %                1e-9)
    %       maxIter: maximum number of calls to funObj (default: 500)
    %       numDiff: compute derivatives numerically (0: use user-supplied
    %       derivatives (default), 1: use finite differences, 2: use
    %                                 complex differentials)
    %       suffDec: sufficient decrease parameter in Armijo condition
    %       (default: 1e-4)
    %       interp: type of interpolation (0: step-size halving, 1:
    %       quadratic, 2: cubic)
    %       memory: number of steps to look back in non-monotone Armijo
    %       condition
    %       useSpectral: use spectral scaling of gradient direction
    %       (default: 1)
    %       projectLS: backtrack along projection Arc (default: 0)
    %       testOpt: test optimality condition (default: 1)
    %       feasibleInit: if 1, then the initial point is assumed to be
    %       feasible
    %       bbType: type of Barzilai Borwein step (default: 1)
    %
    %   Notes:
    %       - if the projection is expensive to compute, you can reduce the
    %           number of projections by setting self.testOpt to 0
    
    
    properties (SetAccess = private, Hidden = false)
        % Norm of projected gradient at x
        pgNorm;
        % Iteration counter
        iter;
        % Execution time
        solveTime;
        % Exit flag
        iStop;
        % Convergence flag
        solved;
        % Projection calls counter
        nProj;
        % Objective function calls counter
        nObjFunc;
    end % gettable private properties
    
    properties (Access = private, Hidden = false)
        % Internal parameters
        verbose; % 0, 1 or 2
        progTol;
        maxEval;
        suffDec;
        memory;
        useSpectral;
        projectLS;
        testOpt;
        bbType;
        maxIterLS;
        fid;
    end % private properties
    
    properties (Hidden = true, Constant)
        LOG_HEADER = { ...
            'Iteration', 'FunEvals', 'Projections', 'Step Length', ...
            'Function Val'};
        LOG_FORMAT = '%10s %10s %10s %15s %15s\n';
        LOG_BODY = '%10d %10d %10d %15.5e %15.5e\n';
        LOG_HEADER_OPT = { ...
            'Iteration', 'FunEvals', 'Projections', 'Step Length', ...
            'Function Val', '||Pg||'};
        LOG_FORMAT_OPT = '%10s %10s %10s %15s %15s %15s\n';
        LOG_BODY_OPT = '%10d %10d %10d %15.5e %15.5e %15.5e\n';
        EXIT_MSG = { ...
            ['First-Order Optimality Conditions Below optTol at', ...
            ' Initial Point\n'], ...                                    % 1
            'Directional Derivative below progTol\n', ...               % 2
            'First-Order Optimality Conditions Below optTol\n', ...     % 3
            'Step size below progTol\n', ...                            % 4
            'Function value changing by less than progTol\n', ...       % 5
            'Function Evaluations exceeds maxEval\n', ...               % 6
            'Maximum number of iterations reached\n'};                  % 7
    end % constant properties
    
    
    methods (Access = public)
        
        function self = SpgSolver(nlp, varargin)
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
            
            self = self@solvers.NlpSolver(nlp, varargin{:});
            
            % Gathering optional arguments and setting default values
            p = inputParser;
            p.KeepUnmatched = false;
            p.addParameter('verbose', 2);
            p.addParameter('progTol', 1e-9);
            p.addParameter('maxEval', 5e2);
            p.addParameter('suffDec', 1e-4);
            p.addParameter('memory', 10);
            p.addParameter('useSpectral', 1);
            p.addParameter('projectLS', 0);
            p.addParameter('testOpt', 1);
            p.addParameter('bbType', 1);
            p.addParameter('fid', 1);
            p.addParameter('maxIterLS', 10); % Max iters for linesearch
            
            p.parse(varargin{:});
            
            self.verbose = p.Results.verbose;
            self.progTol = p.Results.progTol;
            self.maxEval = p.Results.maxEval;
            self.suffDec = p.Results.suffDec;
            self.memory = p.Results.memory;
            self.useSpectral = p.Results.useSpectral;
            self.projectLS = p.Results.projectLS;
            self.testOpt = p.Results.testOpt;
            self.bbType = p.Results.bbType;
            self.maxIterLS = p.Results.maxIterLS;
            self.fid = p.Results.fid;
        end % constructor
        
        function self = solve(self)
            %% Solve using MinConf_SPG
            
            self.solveTime = tic;
            
            % Output Log
            if self.verbose >= 2
                % Printing header
                self.printHeaderFooter('header');
                if self.testOpt
                    self.printf(self.LOG_FORMAT_OPT, ...
                        self.LOG_HEADER_OPT{:});
                else
                    self.printf(self.LOG_FORMAT, ...
                        self.LOG_HEADER{:});
                end
            end
            
            % Exit flag set to 0, will exit if not 0
            self.iStop = 0;
            % Resetting the counters
            self.nProj = 0;
            self.nObjFunc = 0;
            self.iter = 1;
            
            % Evaluate Initial Point
            x = self.project(self.nlp.x0);
            [f, g] = self.obj(x);
            
            self.rOptTol = self.aOptTol * norm(g);
            
            % Optionally check optimality
            pgnrm = 0;
            if self.testOpt
                pgnrm = norm(self.gpstep(x, g));
                if pgnrm < self.rOptTol + self.aOptTol
                    self.iStop = 1; % will bypass main loop
                end
            end
            
            %% Main loop
            while self.iStop == 0
                % Compute Step Direction
                if self.iter == 1 || ~self.useSpectral
                    alph = 1;
                else
                    y = g - gOld;
                    s = x - xOld;
                    if self.bbType == 1
                        alph = (s' * s) / (s' * y);
                    else
                        alph = (s' * y) / (y' * y);
                    end
                    if alph <= 1e-10 || alph > 1e10 || isnan(alph)
                        alph = 1;
                    end
                end
                
                % Descent direction
                d = -alph * g;
                fOld = f;
                xOld = x;
                gOld = g;
                
                % Compute Projected Step
                if ~self.projectLS
                    d = self.gpstep(x, -d); % project(x + d), d = -alph*g
                end
                
                % Check that Progress can be made along the direction
                gtd = g' * d;
                if gtd > -self.progTol * norm(g) * norm(d)
                    self.iStop = 2;
                    % Leaving now saves some processing
                    break;
                end
                
                % Select Initial Guess to step length
                if self.iter == 1
                    t = min(1, 1 / sum(abs(g)));
                else
                    t = 1;
                end
                
                % Compute reference function for non-monotone condition
                if self.memory == 1
                    funRef = f;
                else
                    if self.iter == 1
                        fOldVals = repmat(-inf, [self.memory 1]);
                    end
                    if self.iter <= self.memory
                        fOldVals(self.iter) = f;
                    else
                        fOldVals = [fOldVals(2:end); f];
                    end
                    funRef = max(fOldVals);
                end
                
                % Evaluate the Objective and Gradient at the Initial Step
                if self.projectLS
                    xNew = self.project(x + t * d);
                else
                    xNew = x + t * d;
                end
                
                [fNew, gNew] = self.obj(xNew);
                
                [xNew, fNew, gNew, t, ~] = self.backtracking(x, ...
                    xNew, f, fNew, g, gNew, d, t, funRef);
                
                % Take Step
                x = xNew;
                f = fNew;
                g = gNew;
                
                if self.testOpt
                    pgnrm = norm(self.gpstep(x, g));
                    % Output Log with opt. cond.
                    if self.verbose >= 2
                        fprintf(self.LOG_BODY_OPT, self.iter, ...
                            self.nObjFunc, self.nProj, t, f, pgnrm);
                    end
                else
                    % Output Log without opt. cond.
                    if self.verbose >= 2
                        fprintf(self.LOG_BODY, self.iter, ...
                            self.nObjFunc, self.nProj, t, f);
                    end
                end
                
                % Check optimality
                if self.testOpt
                    if pgnrm < self.rOptTol + self.aOptTol
                        self.iStop = 3;
                    end
                elseif max(abs(t * d)) < self.progTol * norm(d)
                    self.iStop = 4;
                elseif abs((f - fOld)/max([fOld, f, 1])) < self.progTol
                    self.iStop = 5;
                elseif self.nObjFunc > self.maxEval
                    self.iStop = 6;
                elseif self.iter >= self.maxIter
                    self.iStop = 7;
                end
                
                if self.iStop ~= 0
                    break;
                end
                self.x = x;
                self.fx = f;
                self.pgNorm = pgnrm;
                self.iter = self.iter + 1;
            end % main loop
            
            %% End of solve
            self.solved = ~(self.iStop == 6 || self.iStop == 7);
            self.solveTime = toc(self.solveTime);
            if self.verbose
                self.printf('\nEXIT SPG: %s\nCONVERGENCE: %d\n', ...
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
                    self.printf([repmat('\t', 1, 3), 'MinConf_SPG \n']);
                    self.printf('%s\n\n', ['*', repmat('-',1,58), '*']);
                    self.printf(self.nlp.formatting())
                    self.printf('\nParameters\n----------\n')
                    self.printf('%-15s: %3s %8d', 'maxIter', '', ...
                        self.maxIter);
                    self.printf('\t%-15s: %3s %8d\n', ' maxEval', '', ...
                        self.maxEval);
                    self.printf('%-15s: %3s %8.1e', 'optTol', '', ...
                        self.aOptTol);
                    self.printf('\t%-15s: %3s %8d\n', ' useSpectral', ...
                        '', self.useSpectral);
                    self.printf('%-15s: %3s %8.1e', 'suffDec', '', ...
                        self.suffDec);
                    self.printf('\t%-15s: %3s %8.1e\n', ' progTol', '', ...
                        self.progTol);
                    self.printf('%-15s: %3s %8d', 'bbType', '', ...
                        self.bbType);
                    self.printf('\t%-15s: %3s %8d\n', ' memory', '', ...
                        self.memory);
                    self.printf('\n%15s: %3s %8s\n', 'Projection type', ...
                        '', class(self.nlp.projModel));
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
                        self.nlp.ncalls_hvp);
                    self.printf('\n');
                    tt = self.solveTime;
                    t1 = self.nlp.time_fobj + self.nlp.time_fcon;
                    t1t = round(100 * t1/tt);
                    t2 = self.nlp.time_gobj + self.nlp.time_gcon;
                    t2t = round(100 * t2/tt);
                    self.printf([' %-24s %6.2f (%3d%%)  %-20s %6.2f', ...
                        '(%3d%%)\n'], 'Time: function evals' , t1, t1t, ...
                        'gradient evals', t2, t2t);
                    t1 = self.nlp.time_hvp; t1t = round(100 * t1/tt);
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
        
        function s = gpstep(self, x, g)
            %% GPStep - computing the projected gradient
            % Inputs:
            %   - x: current point
            %   - g: gradient at x
            % Output:
            %   - s: projected gradient
            % Calling project to increment projection counter
            s = self.project(x - g) - x;
        end
        
        function [f, g, H] = obj(self, x)
            %% Obj - Evaluate objective function, gradient and hessian at x
            % Input:
            %   - x: current point
            % Ouputs (variable):
            %   - f: value of the objective function at x
            %   - g: gradient of the objective function at x
            %   - H: hessian of the objective function at x
            
            if nargout == 1
                f = self.nlp.obj(x);
            elseif nargout == 2
                [f, g] = self.nlp.obj(x);
            elseif nargout == 3
                [f, g, H] = self.nlp.obj(x);
            end
            self.nObjFunc = self.nObjFunc + 1;
        end
        
        function z = project(self, x)
            %% Project - projecting x on the constraint set
            z = self.nlp.project(x);
            self.nProj = self.nProj + 1;
        end
        
        function [xNew, fNew, gNew, t, failed] = backtracking(self, ...
                x, xNew, f, fNew, g, gNew, d, t, funRef)
            % Backtracking Line Search
            failed = false;
            iterLS = 1;
            while fNew > funRef + self.suffDec* g' * (xNew - x)
                if self.verbose == 2
                    fprintf('Halving Step Size\n');
                end
                t = t / 2;
                
                % Check whether step has become too small
                if max(abs(t * d)) < self.progTol * norm(d) ...
                        || t == 0 || iterLS > self.maxIterLS
                    if self.verbose == 2
                        fprintf('Line Search failed\n');
                    end
                    failed = true;
                    t = 0;
                    xNew = x;
                    fNew = f;
                    gNew = g;
                    return;
                end
                
                if self.projectLS
                    % Projected linesearch
                    xNew = self.project(x + t * d);
                else
                    xNew = x + t * d;
                end
                
                [fNew, gNew] = self.obj(xNew);
                iterLS = iterLS + 1;
            end
        end % backtracking
        
    end % private methods
    
end % class