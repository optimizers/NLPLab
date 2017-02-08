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
        % Projection calls counter
        nProj;
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
            'Directional Derivative below feasTol\n', ...               % 2
            'First-Order Optimality Conditions Below optTol\n', ...     % 3
            'Step size below progTol\n', ...                            % 4
            'Function value changing by less than feasTol\n', ...       % 5
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
            
            % Gathering optional arguments and setting default values
            p = inputParser;
            p.PartialMatching = false;
            p.KeepUnmatched = true;
            p.addParameter('suffDec', 1e-4);
            p.addParameter('memory', 10);
            p.addParameter('useSpectral', 1);
            p.addParameter('projectLS', 0);
            p.addParameter('testOpt', 1);
            p.addParameter('bbType', 0);
            p.addParameter('fid', 1);
            p.addParameter('maxIterLS', 50); % Max iters for linesearch
            
            p.parse(varargin{:});
            
            self = self@solvers.NlpSolver(nlp, p.Unmatched);
            
            self.suffDec = p.Results.suffDec;
            self.memory = p.Results.memory;
            self.useSpectral = p.Results.useSpectral;
            self.projectLS = p.Results.projectLS;
            self.testOpt = p.Results.testOpt;
            self.bbType = p.Results.bbType;
            self.maxIterLS = p.Results.maxIterLS;
            self.fid = p.Results.fid;
            
            import utils.PrintInfo;
            import linesearch.nmArmijo;
        end % constructor
        
        function self = solve(self)
            %% Solve using MinConf_SPG
            
            self.solveTime = tic;
            
            printObj = utils.PrintInfo('Spg');
            
            % Output Log
            if self.verbose >= 2
                % Printing header
                extra = containers.Map( ...
                    {'suffDec', 'memory', 'useSpectral', 'projectLS', ...
                    'testOpt', 'bbType', 'maxIterLS'}, ...
                    {self.suffDec, self.memory, self.useSpectral, ...
                    self.projectLS, self.testOpt, self.bbType, ...
                    self.maxIterLS});
                printObj.header(self, extra);
                
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
            self.iter = 1;
            
            % Evaluate Initial Point
            x = self.project(self.nlp.x0);
            [f, g] = self.nlp.obj(x);
            
            % Relative stopping tolerance
            self.rOptTol = self.aOptTol * norm(g);
            self.rFeasTol = self.aFeasTol * abs(f);
            
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
                if gtd > -self.aFeasTol * norm(g) * norm(d) - self.aFeasTol
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
                if self.memory <= 1
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
                
                [fNew, gNew] = self.nlp.obj(xNew);
                
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
                        self.nObjFunc = self.nlp.ncalls_fobj + ...
                            self.nlp.ncalls_fcon;
                        fprintf(self.LOG_BODY_OPT, self.iter, ...
                            self.nObjFunc, self.nProj, t, f, pgnrm);
                    end
                else
                    % Output Log without opt. cond.
                    if self.verbose >= 2
                        self.nObjFunc = self.nlp.ncalls_fobj + ...
                            self.nlp.ncalls_fcon;
                        fprintf(self.LOG_BODY, self.iter, ...
                            self.nObjFunc, self.nProj, t, f);
                    end
                end
                
                % Check optimality
                if self.testOpt
                    if pgnrm < self.rOptTol + self.aOptTol
                        self.iStop = 3;
                    end
                end
                if max(abs(t * d)) < self.aFeasTol * norm(d) + ...
                        self.aFeasTol
                    self.iStop = 4;
                elseif abs(f - fOld) < self.rFeasTol + self.aFeasTol
                    self.iStop = 5;
                elseif self.nObjFunc > self.maxEval
                    self.iStop = 6;
                elseif self.iter >= self.maxIter
                    self.iStop = 7;
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
                if max(abs(t * d)) < self.aFeasTol * norm(d) ...
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
                
                [fNew, gNew] = self.nlp.obj(xNew);
                iterLS = iterLS + 1;
            end
        end % backtracking
        
    end % private methods
    
end % class