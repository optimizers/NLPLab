classdef CflashSolver < solvers.NlpSolver
    %% CflashSolver - Calls the Cflash solver
    % TRON algorithm generalized to linear inequalities
    % This code is an adaptation of the original bcflash.m project to the
    % case of linear inequalities.
    %
    % The supplied model must be a subclass of NLP model that contains the
    % functions that are required to project on the constraint set, which
    % are
    %       * project
    %       * eqProject
    %
    % A public attribute named
    %
    %       * normJac
    %
    % must also be provided. It must represent the norm of the Jacobian of
    % the constraint set. It is used in the evaluation of relative
    % tolerances.
    %
    % This algorithm also has embedded logging, through the use of the
    % loggin4matlab package, available at
    % https://github.com/optimizers/logging4matlab
    % The repository should be located under the parent repository of the
    % current folder.
    
    
    properties (SetAccess = private, Hidden = false)
        pgNorm % norm of projected gradient at x
        maxCgIter      % maximum number of CG iterations per Newton step
        nSuccessIter = 0 % number of successful iterations
        cgIter = 0    % total number of CG iterations
        gNorm0         % norm of the gradient at x0
        verbose        % log level
        iStop          % exit flag
        exitMsg       % string indicating exit
        backtrack      % Armijo linesearch type backtracking
        eqTol          % Tolerance for equalities (see indFree)
        nProj;         % # of projections
        maxProj;       % maximal # of projections
    end % gettable private properties
    
    properties (SetAccess = private, Hidden = true)
        mu0            % sufficient decrease parameter
        cgTol
        fMin
        fid            % File ID of where to direct log output
    end % private properties
    
    properties (Hidden = true, Constant)
        EXIT_NONE                  = 0;
        EXIT_OPTIMAL               = 1;
        EXIT_ITERATIONS            = 2;
        EXIT_UNBOUNDED             = 3;
        EXIT_FATOL                 = 4;
        EXIT_FRTOL                 = 5;
        EXIT_UNKNOWN               = 6;
        EXIT_PROJECTIONS           = 7;
        EXIT_MSG = {
            'Optimal solution found', ...                               % 1
            'Too many iterations', ...                                  % 2
            'Unbounded below', ...                                      % 3
            'Absolute function tolerance', ...                          % 4
            'Relative function tolerance', ...                          % 5
            'Unknown exit', ...                                         % 6
            'Maximum number of projections reached', ...                % 7
            };
        
        % Constants used to manipulate the TR radius. These are the numbers
        % used by TRON.
        sig1 = 0.25;
        sig2 = 0.50;
        sig3 = 4.00;
        eta0 = 1e-4;
        eta1 = 0.25;
        eta2 = 0.75;
        
        % Log header and body formats.
        logH = '\n%5s  %13s  %13s  %9s %5s  %9s  %9s %9s\n';
        logB = '%5i  %13.6e  %13.6e  %9d %5i  %9.3e  %9.3e  %3s %d\n';
        logT = {'iter', 'f(x)', '|g(x)|', '# Proj', 'cg', 'preRed', ...
            'radius', '#free'};
        
        %         % TRPCG header and body formats
        %         trpcg_header = sprintf('\t%-5s  %9s  %8s  %8s', 'Iter', ...
        %             '||r''*g||', 'curv', '||(Cp)_i||');
        %         trpcg_fmt = '\t%-5d  %9.2e  %8.2e  %8.2e';
        
    end % constant properties
    
    
    methods (Access = public)
        
        function self = CflashSolver(nlp, varargin)
            %% Constructor
            % Inputs:
            %   - nlp: subclass of NLP model that contains the following
            %   projection functions:
            %           * project(self, x)
            %           * eqProject(self, x, ind)
            %   where x is a vector to project and ind represent the active
            %   constraints at x.
            %   - optional arguments: parameters of the Cflash solver, can
            %   be a list of arguments or a struct
            
            % Verifying if the model has the required methods & props
            if ~ismethod(nlp, 'project')
                error('No project method in nlp');
            elseif ~ismethod(nlp, 'eqProject')
                error('No eqProject method in nlp');
            elseif ~isprop(nlp, 'normJac')
                error('No normJac attribute in nlp');
            end
            
            % Parse input parameters and initialize local variables.
            p = inputParser;
            p.PartialMatching = false;
            p.KeepUnmatched = true;
            p.addParameter('maxCgIter', length(nlp.x0));
            p.addParameter('cgTol', 0.1);
            p.addParameter('fMin', -1e32);
            p.addParameter('mu0', 0.01);
            p.addParameter('verbose', 2);
            p.addParameter('fid', 1);
            p.addParameter('backtrack', false);
            p.addParameter('eqTol', 1e-6);
            p.addParameter('maxProj', 1e5);
            
            p.parse(varargin{:});
            
            self = self@solvers.NlpSolver(nlp, p.Unmatched);
            
            % Store various objects and parameters.
            self.cgTol = p.Results.cgTol;
            self.maxCgIter = p.Results.maxCgIter;
            self.fMin = p.Results.fMin;
            self.mu0 = p.Results.mu0;
            self.verbose = p.Results.verbose;
            self.fid = p.Results.fid;
            self.backtrack = p.Results.backtrack;
            self.eqTol = p.Results.eqTol;
            self.maxProj = p.Results.maxProj;
        end % constructor
        
        function self = solve(self)
            %% Solve
            self.solveTime = tic;
            
            % Make sure initial point is feasible
            x = self.project(self.nlp.x0);
            % First objective and gradient evaluation.
            [f, g] = self.nlp.obj(x);
            
            % Initialize stopping tolerance and initial TR radius.
            gNorm = norm(g);
            delta = gNorm;
            self.gNorm0 = gNorm;
            self.rOptTol = self.aOptTol * gNorm;
            self.rFeasTol = self.aFeasTol * abs(f);
            
            % Actual and predicted reductions. Initial inf value prevents
            % exits based on related on first iter.
            actRed = inf;
            preRed = inf;
            
            % Miscellaneous iter
            alphc = 1;
            iterCg = 0;
            sigma1 = self.sig1;
            sigma2 = self.sig2;
            sigma3 = self.sig3;
            self.iStop = self.EXIT_NONE;
            self.nProj = 0;
            
            %% Main loop
            while true
                % Check stopping conditions
                [~, nFree] = self.indFree(x);
                pgNorm = norm(self.gpstep(x, -1, g));
                
                exit = pgNorm <= self.rOptTol + self.aOptTol;
                if ~self.iStop && exit
                    self.iStop = self.EXIT_OPTIMAL;
                end
                
                exit = f < self.fMin;
                if ~self.iStop && exit
                    self.iStop = self.EXIT_UNBOUNDED;
                end
                
                exit = abs(actRed) <= self.aFeasTol && ...
                    preRed  <= self.aFeasTol;
                if ~self.iStop && exit
                    self.iStop = self.EXIT_FATOL;
                end
                
                exit = abs(actRed) <= self.rFeasTol && ...
                    preRed  <= self.rFeasTol;
                if ~self.iStop && exit
                    self.iStop = self.EXIT_FRTOL;
                end
                
                exit = self.iter >= self.maxIter;
                if ~self.iStop && exit
                    self.iStop = self.EXIT_ITERATIONS;
                end
                
                exit = self.nProj >= self.maxProj;
                if ~self.iStop && exit
                    self.iStop = self.EXIT_PROJECTIONS;
                end
                
                % Print current iter to log
                if self.verbose >= 2
                    if mod(self.iter, 20) == 0
                        fprintf(self.logH, self.logT{:});
                        %                         self.logger.info(sprintf(self.logH, self.logT{:}));
                    end
                    if self.iter == 0 || successful
                        status = '';
                    else
                        status = 'rej';
                    end
                    fprintf(self.logB, self.iter, f, pgNorm, ...
                        self.nProj, iterCg, preRed, delta, status, nFree);
                    %                     self.logger.info(sprintf(self.logB, self.iter, ...
                    %                         f, pgNorm, iterCg, preRed, delta, status, ...
                    %                         nFree));
                end
                
                % Act on exit conditions
                if self.iStop
                    self.exitMsg = self.EXIT_MSG{self.iStop};
                    self.x = x;
                    self.fx = f;
                    self.pgNorm = pgNorm;
                    break
                end
                
                %% Iteration starts here
                self.iter = self.iter + 1;
                fc = f;
                xc = x;
                
                % Hessian operator.
                Aprod = @(v)self.nlp.hobjprod(x,zeros(0,1),v);
                
                [alphc, s] = self.cauchy(Aprod, x, g, delta, alphc);
                
                % Projected Newton step.
                [x, s, iterCg, ~] = self.spcg(Aprod, x, g, delta, ...
                    self.cgTol, s, self.maxCgIter);
                self.cgIter = self.cgIter + iterCg;
                
                % Predicted reduction.
                As = self.nlp.hobjprod(x, zeros(0,1), s);
                preRed = -(s' * g + 0.5 * s' * As);
                
                % Compute the objective at this new point.
                f = self.nlp.fobj(x);
                actRed = fc - f;
                snorm = norm(s);
                
                % Update the trust-region radius.
                if self.nSuccessIter == 0
                    delta = min(delta, snorm);
                end
                gts = g' * s;
                if f - fc - gts <= 0
                    alph = sigma3;
                else
                    alph = max(sigma1, -0.5 * gts / (f - fc - gts));
                end
                
                % Changing delta according to a set of rules:
                if actRed < self.eta0 * preRed || actRed == -inf;
                    delta = min(max(alph, sigma1) * snorm, ...
                        sigma2 * delta);
                elseif actRed < self.eta1 * preRed
                    delta = max(sigma1 * delta, min(alph * snorm, ...
                        sigma2 * delta));
                elseif actRed < self.eta2 * preRed
                    delta = max(sigma1 * delta, min(alph * snorm, ...
                        sigma3 * delta));
                else
                    delta = max(delta, min(alph * snorm, sigma3 * delta));
                end
                
                if actRed > self.eta0 * preRed;
                    successful = true;
                elseif self.backtrack
                    % Enter Armijo linesearch if backtracking is enabled
                    [x, armijoStep, info] = self.armijoLineSearch(xc, s);
                    if info
                        % Armijo linesearch was successful
                        f = self.nlp.fobj(x);
                        s = armijoStep * s;
                        armAs = self.nlp.hobjprod(x, zeros(0,1), s);
                        armRed = -(s' * g + 0.5 * s' * armAs);
                        
                        if (fc - f) > self.eta0 * armRed
                            successful = true;
                            delta = armijoStep * snorm;
                            actRed = fc - f;
                            preRed = armRed;
                        else
                            successful = false;
                        end
                    else
                        % Failure of the Armijo linesearch
                        successful = false;
                    end
                else
                    % The step is rejected
                    successful = false;
                end
                if successful
                    self.nSuccessIter = self.nSuccessIter + 1;
                    % Update the gradient value
                    g = self.nlp.gobj(x);
                else
                    f = fc;
                    x = xc;
                end
            end % main loop
            
            self.nObjFunc = self.nlp.ncalls_fobj + self.nlp.ncalls_fcon;
            self.nGrad = self.nlp.ncalls_gobj + self.nlp.ncalls_gcon;
            self.nHess = self.nlp.ncalls_hvp + self.nlp.ncalls_hes;
            
            %% End of solve
            self.solved = ~(self.istop == 2 || self.istop == 6 || ...
                self.istop == 7);
            self.solveTime = toc(self.solveTime);
            if self.verbose
                self.printf('\nEXIT PQN: %s\nCONVERGENCE: %d\n', ...
                    self.EXIT_MSG{self.istop}, self.solved);
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
                    self.printf('%s\n',repmat('=',1,80));
                    self.printf('Constrained FLASH \n');
                    self.printf('%s\n\n',repmat('=',1,80));
                    self.printf(self.nlp.formatting())
                    self.printf('\nParameters\n----------\n')
                    self.printf('%-15s: %3s %8i'  , 'iter max', '', ...
                        self.maxIter);
                    self.printf('%5s','');
                    self.printf('%-15s: %3s %8.1e\n', 'aFeasTol', '', ...
                        self.aFeasTol);
                    self.printf('%-15s: %3s %8.1e'  , 'rFeasTol', '', ...
                        self.rFeasTol);
                    self.printf('%5s','');
                    self.printf('%-15s: %3s %8.1e\n', 'fMin', '', ...
                        self.fMin);
                    self.printf('%-15s: %3s %8.1e', 'cgTol', '', ...
                        self.cgTol);
                    self.printf('%5s','');
                    self.printf('%-15s: %3s %8.1e\n', 'aOptTol', '', ...
                        self.aOptTol);
                    self.printf('%-15s: %3s %8.1e', 'mu0', '', self.mu0);
                    self.printf('%5s', '');
                    self.printf('%-15s: %3s %8i\n', 'maxProj', '', ...
                        self.maxProj);
                case 'footer'
                    % Print footer
                    self.printf('\n');
                    self.printf(' %-27s  %6i     %-17s  %15.8e\n', ...
                        'No. of iterations', self.iter, ...
                        'Objective value', self.fx);
                    t1 = self.nlp.ncalls_fobj + self.nlp.ncalls_fcon;
                    t2 = self.nlp.ncalls_gobj + self.nlp.ncalls_gcon;
                    self.printf(' %-27s  %6i     %-17s    %6i\n', ...
                        'No. of calls to objective' , t1, ...
                        'No. of calls to gradient', t2);
                    self.printf(' %-27s  %6i     %-22s  %10.2e\n',...
                        'No. of Hessian-vector prods', ...
                        self.nlp.ncalls_hvp, ...
                        'No. of successful iterations', ...
                        self.nSuccessIter);
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
        
        function xProj = project(self, x)
            %% Project - simple wrapper to increment nProj counter
            xProj = self.nlp.project(x);
            self.nProj = self.nProj + 1;
        end
        
        function xProj = eqProject(self, x, ind)
            %% EqProject - simple wrapper to increment nProj counter
            xProj = self.nlp.eqProject(x, ind);
            self.nProj = self.nProj + 1;
        end
        
        function [alph, s] = cauchy(self, Aprod, x, g, delta, alph)
            %% CAUCHY
            %
            % This subroutine computes a Cauchy step that satisfies a trust
            % region constraint and a sufficient decrease condition.
            %
            % The Cauchy step is computed for the quadratic
            %
            %       q(s) = 0.5*s'*A*s + g'*s,
            %
            % where A is a symmetric matrix in compressed row storage, and
            % g is a vector. Given a parameter alph, the Cauchy step is
            %
            %       s[alph] = P[x - alph*g] - x,
            %
            % with P the projection onto the constraint set.
            % The Cauchy step satisfies the trust region constraint and the
            % sufficient decrease condition
            %
            %       || s || <= delta,      q(s) <= mu_0*(g'*s),
            %
            % where mu_0 is a constant in (0,1).
            self.logger.debug('-- Entering Cauchy --');
            self.logger.debug(sprintf('α = %7.1e, δ = %7.3e', alph, delta));
            interpf =  0.1;         % interpolation factor
            extrapf = 1 / interpf;  % extrapolation factor
            
            % Find the minimal and maximal break-point on x - alph*g.
            [~, ~, brptMax] = self.breakpt(x, -g);
            self.logger.debug(sprintf('brptMax = %7.1e', brptMax));
            
            % Evaluate the initial alph and decide if the algorithm
            % must interpolate or extrapolate.
            s = self.gpstep(x, -alph, g);
            sNorm = norm(s);
            self.logger.debug(sprintf('||s|| = %7.3e', sNorm));
            if sNorm >= delta
                interp = true;
            else
                wa = Aprod(s);
                gts = g' * s;
                q = 0.5 * s' * wa + gts;
                interp = (q >= self.mu0 * gts);
            end
            
            % Either interpolate or extrapolate to find a successful step.
            if interp
                self.logger.debug('Interpolating');
                % Reduce alph until a successful step is found.
                search = true;
                while search
                    % This is a crude interpolation procedure that
                    % will be replaced in future versions of the code.
                    alph = interpf * alph;
                    s = self.gpstep(x, -alph, g);
                    sNorm = norm(s);
                    self.logger.debug(sprintf('\t||s|| = %7.3e', sNorm));
                    if sNorm <= delta
                        wa = Aprod(s);
                        gts = g' * s;
                        q = 0.5 * s' * wa + gts;
                        search = (q >= self.mu0 * gts);
                    end
                end
            else
                self.logger.debug('Extrapolating');
                % Increase alph until a successful step is found.
                search = true;
                alphas = alph;
                while search && alph <= brptMax
                    % This is a crude extrapolation procedure that
                    % will be replaced in future versions of the code.
                    alph = extrapf * alph;
                    s = self.gpstep(x, -alph, g);
                    sNorm = norm(s);
                    self.logger.debug(sprintf('\t||s|| = %7.3e', sNorm));
                    if sNorm <= delta
                        wa = Aprod(s);
                        gts = g' * s;
                        q = 0.5 * s' * wa + gts;
                        if q <= self.mu0 * gts
                            search = true;
                            alphas = alph;
                        end
                    else
                        search = false;
                    end
                end
                % Recover the last successful step.
                alph = alphas;
                s = self.gpstep(x, -alph, g);
            end
            self.logger.debug(sprintf('Leaving Cauchy, α = %7.1e', alph));
        end % cauchy
        
        function [x, w] = prsrch(self, Aprod, x, g, w)
            %% PRSRCH  Projected search.
            %
            % [x, w] = prsrch(Aprod, x, g, w) where
            %
            %     Inputs:
            %     Aprod is a function handle to compute matrix-vector
            %     products
            %     x        current point
            %     g        current gradient
            %     w        search direction
            %     mu0      linesearch parameter
            %     interpf  interpolation parameter
            %
            %     Output:
            %     x is the final point P[x + alph*w]
            %     w is the step s[alph]
            %
            %   This subroutine uses a projected search to compute a step
            %   that satisfies a sufficient decrease condition for the
            %   quadratic
            %
            %           q(s) = 0.5*s'*A*s + g'*s,
            %
            %   where A is a symmetric matrix in compressed column
            %   storage, and g is a vector. Given the parameter alph,
            %   the step is
            %
            %           s[alph] = P[x + alph*w] - x,
            %
            %   where w is the search direction and P the projection onto
            %   the n-dimensional interval [xl,xu]. The final step
            %   s = s[alph] satisfies the sufficient decrease condition
            %
            %           q(s) <= mu_0*(g'*s),
            %
            %   where mu_0 is a constant in (0,1).
            %
            %   The search direction w must be a descent direction for the
            %   quadratic q at x such that the quadratic is decreasing
            %   in the ray  x + alph*w for 0 <= alph <= 1.
            self.logger.debug('-- Entering prsrch --');
            interpf = 0.5; % Interpolation factor
            
            % Set the initial alph = 1 because the quadratic function is
            % decreasing in the ray x + alph*w for 0 <= alph <= 1.
            alph = 1;
            nSteps = 0;
            
            % Find the smallest break-point on the ray x + alph*w.
            [~, brptMin, ~] = self.breakpt(x, w);
            self.logger.debug(sprintf('brptMin = %7.1e', brptMin));
            
            % Reduce alph until the sufficient decrease condition is
            % satisfied or x + alph*w is feasible.
            search = true;
            self.logger.debug('Interpolating');
            while search && alph > brptMin
                % Calculate P[x + alph*w] - x and check the sufficient
                % decrease condition.
                nSteps = nSteps + 1;
                s = self.gpstep(x, alph, w);
                self.logger.debug(sprintf('\t||s|| = %7.3e', norm(s)));
                As = Aprod(s);
                gts = g' * s;
                q = 0.5 * s' * As + gts;
                if q <= self.mu0 * gts
                    search = false;
                else
                    % This is a crude interpolation procedure that
                    % will be replaced in future versions of the code.
                    alph = interpf * alph;
                end
            end
            
            % Force at least one more constraint to be added to the active
            % set if alph < brptMin and the full step is not successful.
            % There is sufficient decrease because the quadratic function
            % is decreasing in the ray x + alph*w for 0 <= alph <= 1.
            if alph < 1 && alph < brptMin
                alph = brptMin;
            end
            
            % Compute the final iterate and step.
            s = self.gpstep(x, alph, w);
            w = s;
            x = self.project(x + alph * w);
        end % prsrch
        
        function [x, s, iters, info] = spcg(self, Aprod, x, g, delta, ...
                rtol, s, iterMax)
            %% SPCG  Minimize a linearly constrained quadratic.
            %
            % This subroutine generates a sequence of approximate
            % minimizers for the subproblem
            %
            %       min { q(x) : cL <= C*x <= cU }.
            %
            % The quadratic is defined by
            %
            %       q(x[0]+s) = 0.5*s'*A*s + g'*s,
            %
            % where x[0] is a base point provided by the user, A is a
            % symmetric matrix in compressed column storage, and g is a
            % vector.
            %
            % At each stage we have an approximate minimizer x[k], and
            % generate a direction p[k] by using a preconditioned conjugate
            % gradient method on the subproblem
            %
            %       min_p   q(x[k]+p)
            %       sc      || p || <= delta
            %               (Cp)_i = 0,         i in A(x) := (Cx)_i = 0
            %
            % Given p[k], the next minimizer x[k+1] is generated by a
            % projected search.
            %
            % The starting point for this subroutine is x[1] = x[0] + s,
            % where x[0] is a base point and s is the Cauchy step.
            %
            % The subroutine converges when the step s satisfies
            %
            %       || (g + A*s)[free] || <= rtol*|| g[free] ||
            %
            % In this case the final x is an approximate minimizer in the
            % face defined by the free variables.
            %
            % The subroutine terminates when the trust region bound does
            % not allow further progress, that is, || p[k] || = delta.
            % In this case the final x satisfies q(x) < q(x[k]).
            %
            % On exit info is set as follows:
            %
            %      info = 1  Convergence. The final step s satisfies
            %                || (g + A*s)[free] || <= rtol*|| g[free] ||,
            %                and the final x is an approximate minimizer
            %                in the face defined by the free variables.
            %
            %      info = 2  Termination. The trust region bound does
            %                not allow further progress.
            %
            %      info = 3  Failure to converge within iterMax iterations.
            self.logger.debug('-- Entering SPCG --');
            
            % Compute A*(x[1] - x[0]) and store in w.
            As = Aprod(s);
            
            % Compute the Cauchy point.
            x = self.project(x + s);
            
            % Start the main iter loop.
            % There are at most n iterations because at each iter
            % at least one variable becomes active.
            iters = 0;
            for nFaces = 1:self.nlp.n
                % Determine the free variables at the current minimizer.
                [indFree, ~] = self.indFree(x);
                
                % Compute the gradient grad q(x[k]) = g + A*(x[k] - x[0]),
                gfNorm = norm(g);
                gfobj = As + g;
                
                % Solve the trust region subproblem in the free variables
                % to generate a direction p[k]. Store p[k] in the array w.
                tol = rtol*gfNorm;
                
                [w, iterTR, infotr] = self.trpcg(Aprod, gfobj, delta, ...
                    tol, iterMax, indFree);
                
                iters = iters + iterTR;
                
                % Use a projected search to obtain the next iterate.
                % The projected search algorithm stores s[k] in w.
                [x, w] = self.prsrch(Aprod, x, gfobj, w);
                
                % Update the minimizer and the step.
                % Note that s now contains x[k+1] - x[0].
                s = s + w;
                
                % Compute A*(x[k+1] - x[0]) and store in w.
                As = Aprod(s);
                
                % Compute the gradient grad
                % q(x[k+1]) = g + A*(x[k+1] - x[0])
                % of q at x[k+1] for the free variables.
                gfobj = As + g;
                gfnormf = norm(gfobj);
                
                % Convergence and termination test.
                % We terminate if the preconditioned conjugate gradient
                % method encounters a direction of negative curvature, or
                % if the step is at the trust region bound.
                if gfnormf <= rtol * gfNorm
                    info = 1;
                    self.logger.debug(sprintf(['Leaving SPCG, info = %d', ...
                        ' (conv)'], info));
                    return
                elseif infotr == 3 || infotr == 4
                    info = 2;
                    self.logger.debug(sprintf(['Leaving SPCG, info = %d', ...
                        ' (TR)'], info));
                    return
                elseif iters > iterMax
                    info = 3;
                    self.logger.debug(sprintf(['Leaving SPCG, info = %d', ...
                        ' (fail)'], info));
                    return
                end
            end % faces
        end % spcg
        
        function s = gpstep(self, x, alph, w)
            %% GPStep - Compute the gradient projection step.
            % s = P[x + alph*w] - x,
            % where P is the projection onto the linear constraint set
            s = self.project(x + alph*w) - x;
        end
        
        function [indFree, nFree] = indFree(self, x)
            %% Indfree - Find indices of the free variables
            % Assuming linear constraints
            % cL <= C*x <= cU
            % where C is the jacobian of the linear constraints
            Cx = self.nlp.fcon(x); % C*x
            % Represents "relative" zero value, smallest approx is eps
            appZero = max(self.eqTol * self.nlp.normJac * norm(x), eps);
            % The equalities cU - C*x = 0 and C*x - cL = 0 are not likely
            % to happen in practice
            indFree = (self.nlp.cU - Cx >= appZero) & ...
                (Cx - self.nlp.cL >= appZero);
            nFree = sum(indFree);
        end
        
        function [w, iters, info] = trpcg(self, Aprod, g, delta, ...
                tol, iterMax, indFree)
            %% TRPCG
            % Given a sparse symmetric matrix A in compressed column
            % storage, this subroutine uses a projected conjugate gradient
            % method to find an approximate minimizer of the trust region
            % subproblem
            %
            %       min_s   q(s)
            %       sc      || s || <= delta
            %               c_i'*s = 0,         for i in A(x):= c_i'*x = 0
            %
            % where q is the quadratic
            %
            %       q(s) = 0.5*s'*A*s + g'*s,
            %
            % Termination occurs if the conjugate gradient iterates leave
            % the trust region, a negative curvature direction is generated
            % or one of the following two convergence tests is satisfied.
            %
            % Convergence in the original variables:
            %
            %       || grad q(s) || <= tol
            %
            % On exit info is set as follows:
            %
            %       info = 1  Convergence in the original variables.
            %                 || grad q(s) || <= tol
            %
            %       info = 2  Convergence in the original variables.
            %                 || grad q(s) || <= tol
            %
            %       info = 3  Negative curvature direction generated.
            %                 In this case || w || = delta and a direction
            %                 of negative curvature w can be recovered by
            %                 solving L'*w = p.
            %
            %       info = 4  Conjugate gradient iterates exit the
            %                 trust region. In this case || w || = delta.
            %
            %       info = 5  Failure to converge within iterMax iterations
            
            self.logger.debug('-- Entering TRPCG --');
            self.logger.debug(sprintf('tol = %7.3e, δ = %7.3e,', tol, delta));
            % Initialize the iterate w and the residual r.
            w = zeros(self.nlp.n, 1);
            % Initialize the residual r of grad q to -g.
            r = -g;
            % Initialize the direction p.
            p = r;
            % Initialize rho and the norms of r.
            rho = r'*r;
            
            % Exit if g = 0.
            if sqrt(rho) == 0
                iters = 0;
                info = 1;
                self.logger.debug(sprintf(['Leaving TRPCG, info', ...
                    '= %d (conv)'], info));
                return
            end
            
            for iters = 1:iterMax
                appZero = max(self.eqTol * self.nlp.normJac * norm(p), ...
                    eps);
                Cp = self.nlp.fcon(p);
                chk = any(Cp(~indFree) >= appZero);
                if chk
                    % Project {p : (C*p)_i = 0} for i such as (C*x)_i = 0
                    p = self.eqProject(p, ~indFree);
                end
                
                %                 Cp = self.nlp.fcon(p); % Computes C*p
                %                 normCp = norm(Cp(~indFree)); % Should be near 0
                %                 self.logger.debug(sprintf('\t||(C*p)_i|| = %7.3e', normCp));
                
                % Compute alph and determine sigma such that the trust
                % region constraint || w + sigma*p || = delta is satisfied.
                q = Aprod(p);
                ptq = p'*q;
                self.logger.debug(sprintf('\tp''*H*p = %7.3e', ptq));
                if ptq > 0
                    alph = rho/ptq;
                else
                    alph = 0;
                end
                
                sigma = solvers.CflashSolver.trqsol(w, p, delta);
                % Exit if there is negative curvature or if the
                % iterates exit the trust region.
                self.logger.debug(sprintf('\tαCG = %7.1e, σ = %7.1e', ...
                    alph, sigma));
                if (ptq <= 0 || alph >= sigma)
                    if sigma ~= 0
                        w = w + sigma*p;
                    end
                    if ptq <= 0
                        info = 3;
                        self.logger.debug(sprintf(['Leaving TRPCG, info', ...
                            ' = %d (negative curv)'], info));
                    else
                        info = 4;
                        self.logger.debug(sprintf(['Leaving TRPCG, info', ...
                            ' = %d (exit TR)'], info));
                    end
                    return
                end
                
                % Update w and the residuals r.
                w = w + alph*p;
                r = r - alph*q;
                % Exit if the residual convergence test is satisfied.
                rtr = r'*r;
                rnorm = sqrt(rtr);
                self.logger.debug(sprintf('\t||r''*r|| = %7.3e', rnorm));
                if rnorm <= tol
                    info = 1;
                    self.logger.debug(sprintf(['Leaving TRPCG, info', ...
                        ' = %d (conv)'], info));
                    return
                end
                % Compute p = r + betaFactor*p and update rho.
                betaFactor = rtr/rho;
                p = r + betaFactor*p;
                rho = rtr;
            end % for loop
            
            iters = iterMax;
            info = 5;
            self.logger.debug(sprintf('Leaving TRPCG, info = %d (fail)', ...
                info));
        end % trpcg
        
        function [nBrpt, brptMin, brptMax] = breakpt(self, x, w)
            %% Breakpt
            % Find the breakpoints on the constraint set
            %                   cL <= C*x <= cU
            % from x in the direction w, i.e. finding alphas such as
            %                   C*(x + alph*w) = cL
            % or
            %                   C*(x + alph*w) = cU
            % NOTE: inequalities set to -inf or inf are deactivated, since
            % they represent an unbounded constraint
            
            Cx = self.nlp.fcon(x); % C*x
            Cw = self.nlp.fcon(w); % C*w
            
            % Smallest approximation is eps
            appZero = max(self.eqTol * self.nlp.normJac * norm(x), eps);
            
            % Redefining iLow & iUpp (lower & upper indices of bounded
            % constraints) as positions instead of logical
            iLow = find(self.nlp.iLow == true);
            iUpp = find(self.nlp.iUpp == true);
            %             iLow = 1:self.nlp.n;
            %             iUpp = iLow;
            
            % Lower constraint intersections: Ax - cL > 0 & Aw < 0
            dec = (Cx(iLow) - self.nlp.cL(iLow) >= appZero) & Cw(iLow) < 0;
            % Upper constraint intersections: cU - Cx > 0 & Aw > 0
            inc = (self.nlp.cU(iUpp) - Cx(iUpp) >= appZero) & Cw(iUpp) > 0;
            
            nBrpt = sum(dec) + sum(inc);
            
            % Quick exit if no breakpoints
            if nBrpt == 0
                brptMin = 0;
                brptMax = 0;
                return
            end
            
            dec = iLow(dec);
            inc = iUpp(inc);
            
            brptDec = (self.nlp.cL(dec) - Cx(dec)) ./ Cw(dec);
            brptInc = (self.nlp.cU(inc) - Cx(inc)) ./ Cw(inc);
            
            % Finding min & max breakpoint
            brptMin =  inf;
            brptMax = -inf;
            
            if any(brptDec)
                brptMinDec = min(brptDec);
                brptMin = min(brptMin, brptMinDec);
                
                brptMaxDec = max(brptDec);
                brptMax = max(brptMax, brptMaxDec);
            end
            if any(brptInc)
                brptMinInc = min(brptInc);
                brptMin = min(brptMin, brptMinInc);
                
                brptMaxInc = max(brptInc);
                brptMax = max(brptMax, brptMaxInc);
            end
        end % breakpt
        
        function [x, alph, info] = armijoLineSearch(self, x, s, varargin)
            %% Armijo Linesearch Backtracking
            % Computes an Armijo linesearch from x in the direction s. This
            % function requires nlp model's obj to get the value of the
            % objective function and the gradient at a point x.
            % Inputs:
            %   - x: initial point of the linesearch
            %   - s: direction of the linesearch
            %   - <varargin>: optional parameters of the method:
            % Outputs:
            %   - x: value obtained by the linesearch procedure
            %   - alph: step length at the end of the procedure
            %   - info:
            %       true : obtained an iterate x that satisfies the Armijo
            %       condition
            %       false : reached number of maximal iterations or minimal
            %       step length
            
            self.logger.debug('-- Entering Armijo linesearch --');
            % Parsing optional arguments
            p = inputParser;
            p.addParameter('alph0', 0.9999);
            p.addParameter('iterMax', 20);
            p.addParameter('sTol', 1e-4);
            p.addParameter('stepTol', 1e-6);
            p.addParameter('tau', 0.5);
            p.parse(varargin{:});
            
            % Retrieving optional arguments
            alph0 = p.Results.alph0;
            % Interpolation factor for the linesearch
            tau = p.Results.tau;
            % Tolerance on the slope
            sTol = p.Results.sTol;
            % Lower bound on the value of the step size
            stepTol = p.Results.stepTol;
            % Maximal number of iterations for the backtracking procedure
            iterMax = p.Results.iterMax;
            
            % Only alph values in ]0, 1[ are allowed
            assert((alph0 < 1) & (alph0 > 0));
            % Only tau values in ]0, 1[ are allowed
            assert((tau < 1) & (tau > 0));
            
            % Iteration counter
            iter = 1;
            
            % Getting objective function and gradient value at x
            [f, g] = self.nlp.obj(x);
            alph = alph0;
            while true
                self.logger.debug(sprintf('\tα = %7.1e', alph));
                % Armijo's condition
                if self.nlp.fobj(x + alph * s) > f + alph * sTol * g' * s
                    % Condition has not been met, reduce alph
                    alph = alph * tau;
                else
                    % Return new value
                    x = x + alph * s;
                    info = true;
                    self.logger.debug(sprintf(['Armijo linesearch', ...
                        ' is successful']));
                    return
                end
                if iter > iterMax || alph < stepTol
                    % Too many iterations or step size too small
                    info = false;
                    self.logger.debug(sprintf(['Failure of the', ...
                        ' Armijo linesearch']));
                    return
                end
                iter = iter + 1;
            end
        end % armijo backtracking
        
    end % private methods
    
    
    methods (Access = public, Hidden = true)
        
        function printf(self, varargin)
            fprintf(self.fid, varargin{:});
        end % function printf
        
    end % public hidden methods
    
    
    methods (Access = private, Static)
        
        function sigma = trqsol(x, p, delta)
            %% TRQSOL  Largest solution of the TR equation.
            %   This subroutine computes the largest (non-negative)
            %   solution of the quadratic trust region equation
            %                   ||x + sigma*p|| = delta.
            %   The code is only guaranteed to produce a non-negative
            %   solution if ||x|| <= delta, and p != 0. If the trust region
            %   equation has no solution, sigma = 0.
            
            ptx = p'*x;
            ptp = p'*p;
            xtx = x'*x;
            dsq = delta^2;
            
            % Guard against abnormal cases.
            rad = ptx^2 + ptp*(dsq - xtx);
            rad = sqrt(max(rad,0));
            
            if ptx > 0
                sigma = (dsq - xtx)/(ptx + rad);
            elseif rad > 0
                sigma = (rad - ptx)/ptp;
            else
                sigma = 0;
            end
        end % trqsol
        
    end % private static methods
    
end % class