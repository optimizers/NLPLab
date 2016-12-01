classdef CFlashSolver < solvers.NLPSolver
    %% CFlashSolver - Calls the CFlash solver
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
        x              % solution
        fx             % objective value at x
        proj_grad_norm % norm of projected gradient at x
        maxiter        % maximum number of iterations
        maxcgiter      % maximum number of CG iterations per Newton step
        iteration = 0  % iteration counter
        num_successful_itns = 0 % number of successful iterations
        cgiters = 0    % total number of CG iterations
        gnorm0         % norm of the gradient at x0
        time_total     % total solve time
        nlp            % copy of the nlp object
        verbose        % log level
        eFlag          % exit flag
        exit_msg       % string indicating exit
        stop_tol       % Stopping tolerance relative to gradient norm
        log            % Added logger (logging4matlab)
        backtrack      % Armijo linesearch type backtracking
        eqTol          % Tolerance for equalities (see indfree)
        nProj;         % # of projections
        maxProj;       % maximal # of projections
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    properties (SetAccess = private, Hidden = true)
        mu0            % sufficient decrease parameter
        gtol
        cgtol
        fatol          % absoulte error in function
        frtol          % relative error in function
        fmin
        fid            % File ID of where to direct log output
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
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
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    methods (Access = public)
        
        function self = CFlashSolver(nlp, varargin)
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
            
            if ~isa(nlp, 'model.nlpmodel')
                error('nlp should be a subclass of nlpmodel');
            end
            % Verifying if the model has the required methods & props
            if ~ismethod(nlp, 'project')
                error('No project method in nlp');
            elseif ~ismethod(nlp, 'eqProject')
                error('No eqProject method in nlp');
            elseif ~isprop(nlp, 'normJac')
                error('No normJac attribute in nlp');
            end
            
            self = self@solvers.NLPSolver(nlp, varargin{:});
            
            % -------------------------------------------------------------
            % Parse input parameters and initialize local variables.
            % -------------------------------------------------------------
            p = inputParser;
            p.addParameter('x0', nlp.x0);
            p.addParameter('maxiter', 10*length(nlp.x0));
            p.addParameter('maxcgiter', length(nlp.x0));
            p.addParameter('cgtol', 0.1);
            p.addParameter('fatol', 0);
            p.addParameter('frtol', 1e-12);
            p.addParameter('gtol', 1e-5);
            p.addParameter('fmin', -1e32);
            p.addParameter('mu0', 0.01);
            p.addParameter('verbose', 1);
            p.addParameter('fid', 1);
            p.addParameter('loggerName', 'tronLogger')
            p.addParameter('logLevel', logging.logging.INFO);
            p.addParameter('backtrack', false);
            p.addParameter('eqTol', 1e-6);
            p.addParameter('maxProj', 1e5);
            p.parse(varargin{:});
            
            % -------------------------------------------------------------
            % Store various objects and parameters.
            % -------------------------------------------------------------
            self.nlp = nlp;
            self.gtol = p.Results.gtol;
            self.cgtol = p.Results.cgtol;
            self.fatol = p.Results.fatol;
            self.frtol = p.Results.frtol;
            self.maxiter = p.Results.maxiter;
            self.maxcgiter = p.Results.maxcgiter;
            self.fmin = p.Results.fmin;
            self.mu0 = p.Results.mu0;
            self.verbose = p.Results.verbose;
            self.fid = p.Results.fid;
            self.log = logging.getLogger(p.Results.loggerName);
            self.log.setCommandWindowLevel(p.Results.logLevel);
            self.backtrack = p.Results.backtrack;
            self.eqTol = p.Results.eqTol;
            self.maxProj = p.Results.maxProj;
            self.nlp.x0 = p.Results.x0;
        end
        
        function self = solve(self)
            %% Solve
            self.time_total = tic;
            
            % Make sure initial point is feasible
            x = self.project(self.nlp.x0);
            % First objective and gradient evaluation.
            [f, g] = self.nlp.obj(x);
            
            % Initialize stopping tolerance and initial TR radius.
            gnorm = norm(g);
            delta = gnorm;
            self.gnorm0 = gnorm;
            self.stop_tol = self.gtol * gnorm;
            
            % Actual and predicted reductions. Initial inf value prevents
            % exits based on related on first iteration.
            actred = inf;
            prered = inf;
            
            % Miscellaneous iteration
            alphac = 1;
            cgits = 0;
            sigma1 = self.sig1;
            sigma2 = self.sig2;
            sigma3 = self.sig3;
            self.eFlag = self.EXIT_NONE;
            self.nProj = 0;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %                      --- Main loop ---
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            while true
                % Check stopping conditions
                [~, nfree] = self.indfree(x);
                pgnorm = norm(self.gpstep(x, -1, g));
                
                exit = pgnorm <= self.stop_tol;
                if ~self.eFlag && exit
                    self.eFlag = self.EXIT_OPTIMAL;
                end
                
                exit = f < self.fmin;
                if ~self.eFlag && exit
                    self.eFlag = self.EXIT_UNBOUNDED;
                end
                
                exit = abs(actred) <= self.fatol && ...
                    prered  <= self.fatol;
                if ~self.eFlag && exit
                    self.eFlag = self.EXIT_FATOL;
                end
                
                exit = abs(actred) <= self.frtol*abs(f) && ...
                    prered  <= self.frtol*abs(f);
                if ~self.eFlag && exit
                    self.eFlag = self.EXIT_FRTOL;
                end
                
                exit = self.iteration >= self.maxiter;
                if ~self.eFlag && exit
                    self.eFlag = self.EXIT_ITERATIONS;
                end
                
                exit = self.nProj >= self.maxProj;
                if ~self.eFlag && exit
                    self.eFlag = self.EXIT_PROJECTIONS;
                end
                
                % ---------------------------------------------------------
                % Print current iteration to log
                % ---------------------------------------------------------
                if self.verbose >= 2
                    if mod(self.iteration, 20) == 0
                        fprintf(self.logH, self.logT{:});
                        %                         self.log.info(sprintf(self.logH, self.logT{:}));
                    end
                    if self.iteration == 0 || successful
                        status = '';
                    else
                        status = 'rej';
                    end
                    fprintf(self.logB, self.iteration, f, pgnorm, ...
                        self.nProj, cgits, prered, delta, status, nfree);
                    %                     self.log.info(sprintf(self.logB, self.iteration, ...
                    %                         f, pgnorm, cgits, prered, delta, status, ...
                    %                         nfree));
                end
                
                % ---------------------------------------------------------
                % Act on exit conditions
                % ---------------------------------------------------------
                if self.eFlag
                    self.exit_msg = self.EXIT_MSG{self.eFlag};
                    self.x = x;
                    self.fx = f;
                    self.proj_grad_norm = pgnorm;
                    break
                end
                
                % ---------------------------------------------------------
                % Iteration starts here
                % ---------------------------------------------------------
                self.iteration = self.iteration + 1;
                fc = f;
                xc = x;
                
                % Hessian operator.
                Aprod = @(v)self.nlp.hobjprod(x,zeros(0,1),v);
                
                [alphac, s] = self.cauchy(Aprod, x, g, delta, alphac);
                
                % Projected Newton step.
                [x, s, cgits, ~] = self.spcg(Aprod, x, g, delta, ...
                    self.cgtol, s, self.maxcgiter);
                self.cgiters = self.cgiters + cgits;
                
                % Predicted reduction.
                As = self.nlp.hobjprod(x, zeros(0,1), s);
                prered = -(s' * g + 0.5 * s' * As);
                
                % Compute the objective at this new point.
                f = self.nlp.fobj(x);
                actred = fc - f;
                snorm = norm(s);
                
                % Update the trust-region radius.
                if self.num_successful_itns == 0
                    delta = min(delta, snorm);
                end
                gts = g' * s;
                if f - fc - gts <= 0
                    alpha = sigma3;
                else
                    alpha = max(sigma1, -0.5 * gts / (f - fc - gts));
                end
                
                % Changing delta according to a set of rules:
                if actred < self.eta0 * prered || actred == -inf;
                    delta = min(max(alpha, sigma1) * snorm, ...
                        sigma2 * delta);
                elseif actred < self.eta1 * prered
                    delta = max(sigma1 * delta, min(alpha * snorm, ...
                        sigma2 * delta));
                elseif actred < self.eta2 * prered
                    delta = max(sigma1 * delta, min(alpha * snorm, ...
                        sigma3 * delta));
                else
                    delta = max(delta, min(alpha * snorm, sigma3 * delta));
                end
                
                if actred > self.eta0 * prered;
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
                            actred = fc - f;
                            prered = armRed;
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
                    self.num_successful_itns = self.num_successful_itns ...
                        + 1;
                    % Update the gradient value
                    g = self.nlp.gobj(x);
                else
                    f = fc;
                    x = xc;
                end
            end % loop
            
            % -------------------------------------------------------------
            % End of solve
            % -------------------------------------------------------------
            self.solved = ~(self.istop == 2 || self.istop == 6 || ...
                self.istop == 7);
            self.time_total = toc(self.time_total);
            if self.verbose
                self.printf('\nEXIT PQN: %s\nCONVERGENCE: %d\n', ...
                    self.EXIT_MSG{self.istop}, self.solved);
                self.printf('||Pg|| = %8.1e\n', self.proj_grad_norm);
                self.printf('Stop tolerance = %8.1e\n', self.stopTol);
            end
            if self.verbose >= 2
                self.printHeaderFooter('footer');
            end
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    methods (Access = private)
        
        function printHeaderFooter(self, msg)
            switch msg
                case 'header'
                    % -----------------------------------------------------
                    % Print header
                    % -----------------------------------------------------
                    self.printf('\n');
                    self.printf('%s\n',repmat('=',1,80));
                    self.printf('Constrained FLASH \n');
                    self.printf('%s\n\n',repmat('=',1,80));
                    self.printf(self.nlp.formatting())
                    self.printf('\nParameters\n----------\n')
                    self.printf('%-15s: %3s %8i'  , 'iter max', '', ...
                        self.maxiter);
                    self.printf('%5s','');
                    self.printf('%-15s: %3s %8.1e\n', 'fatol', '', ...
                        self.fatol);
                    self.printf('%-15s: %3s %8.1e'  , 'frtol', '', ...
                        self.frtol);
                    self.printf('%5s','');
                    self.printf('%-15s: %3s %8.1e\n', 'fmin', '', ...
                        self.fmin);
                    self.printf('%-15s: %3s %8.1e', 'cgtol', '', ...
                        self.cgtol);
                    self.printf('%5s','');
                    self.printf('%-15s: %3s %8.1e\n', 'gtol', '', ...
                        self.gtol);
                    self.printf('%-15s: %3s %8.1e', 'mu0', '', self.mu0);
                    self.printf('%5s', '');
                    self.printf('%-15s: %3s %8i\n', 'maxProj', '', ...
                        self.maxProj);
                case 'footer'
                    % -----------------------------------------------------
                    % Print footer
                    % -----------------------------------------------------
                    self.printf('\n');
                    self.printf(' %-27s  %6i     %-17s  %15.8e\n', ...
                        'No. of iterations', self.iteration, ...
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
                        self.num_successful_itns);
                    self.printf('\n');
                    tt = self.time_total;
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
            end
        end
        
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
        
        function [alpha, s] = cauchy(self, Aprod, x, g, delta, alpha)
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
            % g is a vector. Given a parameter alpha, the Cauchy step is
            %
            %       s[alpha] = P[x - alpha*g] - x,
            %
            % with P the projection onto the constraint set.
            % The Cauchy step satisfies the trust region constraint and the
            % sufficient decrease condition
            %
            %       || s || <= delta,      q(s) <= mu_0*(g'*s),
            %
            % where mu_0 is a constant in (0,1).
            self.log.debug('-- Entering Cauchy --');
            self.log.debug(sprintf('α = %7.1e, δ = %7.3e', alpha, delta));
            interpf =  0.1;         % interpolation factor
            extrapf = 1 / interpf;  % extrapolation factor
            
            % Find the minimal and maximal break-point on x - alpha*g.
            [~, ~, brptmax] = self.breakpt(x, -g);
            self.log.debug(sprintf('brptmax = %7.1e', brptmax));
            
            % Evaluate the initial alpha and decide if the algorithm
            % must interpolate or extrapolate.
            s = self.gpstep(x, -alpha, g);
            sNorm = norm(s);
            self.log.debug(sprintf('||s|| = %7.3e', sNorm));
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
                self.log.debug('Interpolating');
                % Reduce alpha until a successful step is found.
                search = true;
                while search
                    % This is a crude interpolation procedure that
                    % will be replaced in future versions of the code.
                    alpha = interpf * alpha;
                    s = self.gpstep(x, -alpha, g);
                    sNorm = norm(s);
                    self.log.debug(sprintf('\t||s|| = %7.3e', sNorm));
                    if sNorm <= delta
                        wa = Aprod(s);
                        gts = g' * s;
                        q = 0.5 * s' * wa + gts;
                        search = (q >= self.mu0 * gts);
                    end
                end
            else
                self.log.debug('Extrapolating');
                % Increase alpha until a successful step is found.
                search = true;
                alphas = alpha;
                while search && alpha <= brptmax
                    % This is a crude extrapolation procedure that
                    % will be replaced in future versions of the code.
                    alpha = extrapf * alpha;
                    s = self.gpstep(x, -alpha, g);
                    sNorm = norm(s);
                    self.log.debug(sprintf('\t||s|| = %7.3e', sNorm));
                    if sNorm <= delta
                        wa = Aprod(s);
                        gts = g' * s;
                        q = 0.5 * s' * wa + gts;
                        if q <= self.mu0 * gts
                            search = true;
                            alphas = alpha;
                        end
                    else
                        search = false;
                    end
                end
                % Recover the last successful step.
                alpha = alphas;
                s = self.gpstep(x, -alpha, g);
            end
            self.log.debug(sprintf('Leaving Cauchy, α = %7.1e', alpha));
        end % function cauchy
        
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
            %     x is the final point P[x + alpha*w]
            %     w is the step s[alpha]
            %
            %   This subroutine uses a projected search to compute a step
            %   that satisfies a sufficient decrease condition for the
            %   quadratic
            %
            %           q(s) = 0.5*s'*A*s + g'*s,
            %
            %   where A is a symmetric matrix in compressed column
            %   storage, and g is a vector. Given the parameter alpha,
            %   the step is
            %
            %           s[alpha] = P[x + alpha*w] - x,
            %
            %   where w is the search direction and P the projection onto
            %   the n-dimensional interval [xl,xu]. The final step
            %   s = s[alpha] satisfies the sufficient decrease condition
            %
            %           q(s) <= mu_0*(g'*s),
            %
            %   where mu_0 is a constant in (0,1).
            %
            %   The search direction w must be a descent direction for the
            %   quadratic q at x such that the quadratic is decreasing
            %   in the ray  x + alpha*w for 0 <= alpha <= 1.
            self.log.debug('-- Entering prsrch --');
            interpf = 0.5; % Interpolation factor
            
            % Set the initial alpha = 1 because the quadratic function is
            % decreasing in the ray x + alpha*w for 0 <= alpha <= 1.
            alpha = 1;
            nsteps = 0;
            
            % Find the smallest break-point on the ray x + alpha*w.
            [~, brptmin, ~] = self.breakpt(x, w);
            self.log.debug(sprintf('brptmin = %7.1e', brptmin));
            
            % Reduce alpha until the sufficient decrease condition is
            % satisfied or x + alpha*w is feasible.
            search = true;
            self.log.debug('Interpolating');
            while search && alpha > brptmin
                % Calculate P[x + alpha*w] - x and check the sufficient
                % decrease condition.
                nsteps = nsteps + 1;
                s = self.gpstep(x, alpha, w);
                self.log.debug(sprintf('\t||s|| = %7.3e', norm(s)));
                As = Aprod(s);
                gts = g'*s;
                q = 0.5*s'*As + gts;
                if q <= self.mu0*gts
                    search = false;
                else
                    % This is a crude interpolation procedure that
                    % will be replaced in future versions of the code.
                    alpha = interpf*alpha;
                end
            end
            
            % Force at least one more constraint to be added to the active
            % set if alpha < brptmin and the full step is not successful.
            % There is sufficient decrease because the quadratic function
            % is decreasing in the ray x + alpha*w for 0 <= alpha <= 1.
            if alpha < 1 && alpha < brptmin
                alpha = brptmin;
            end
            
            % Compute the final iterate and step.
            s = self.gpstep(x, alpha, w);
            w = s;
            x = self.project(x + alpha*w);
        end % function prsrch
        
        function [x, s, iters, info] = spcg(self, Aprod, x, g, delta, ...
                rtol, s, itermax)
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
            %      info = 3  Failure to converge within itermax iterations.
            self.log.debug('-- Entering SPCG --');
            
            % Compute A*(x[1] - x[0]) and store in w.
            As = Aprod(s);
            
            % Compute the Cauchy point.
            x = self.project(x + s);
            
            % Start the main iteration loop.
            % There are at most n iterations because at each iteration
            % at least one variable becomes active.
            iters = 0;
            for nfaces = 1:self.nlp.objSize
                % Determine the free variables at the current minimizer.
                [indfree, ~] = self.indfree(x);
                
                % Compute the gradient grad q(x[k]) = g + A*(x[k] - x[0]),
                gfnorm = norm(g);
                gfobj = As + g;
                
                % Solve the trust region subproblem in the free variables
                % to generate a direction p[k]. Store p[k] in the array w.
                tol = rtol*gfnorm;
                
                [w, itertr, infotr] = self.trpcg(Aprod, gfobj, delta, ...
                    tol, itermax, indfree);
                
                iters = iters + itertr;
                
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
                if gfnormf <= rtol*gfnorm
                    info = 1;
                    self.log.debug(sprintf(['Leaving SPCG, info = %d', ...
                        ' (conv)'], info));
                    return
                elseif infotr == 3 || infotr == 4
                    info = 2;
                    self.log.debug(sprintf(['Leaving SPCG, info = %d', ...
                        ' (TR)'], info));
                    return
                elseif iters > itermax
                    info = 3;
                    self.log.debug(sprintf(['Leaving SPCG, info = %d', ...
                        ' (fail)'], info));
                    return
                end
                
            end % for faces
        end % function spcg
        
        function s = gpstep(self, x, alpha, w)
            %% GPStep - Compute the gradient projection step.
            % s = P[x + alpha*w] - x,
            % where P is the projection onto the linear constraint set
            s = self.project(x + alpha*w) - x;
        end % function gpstep
        
        function [indfree, nfree] = indfree(self, x)
            %% Indfree - Find indices of the free variables
            % Assuming linear constraints
            % cL <= C*x <= cU
            % where C is the jacobian of the linear constraints
            Cx = self.nlp.fcon(x); % C*x
            % Represents "relative" zero value, smallest approx is eps
            appZero = max(self.eqTol * self.nlp.normJac * norm(x), eps);
            % The equalities cU - C*x = 0 and C*x - cL = 0 are not likely
            % to happen in practice
            indfree = (self.nlp.cU - Cx >= appZero) & ...
                (Cx - self.nlp.cL >= appZero);
            nfree = sum(indfree);
        end
        
        function [w, iters, info] = trpcg(self, Aprod, g, delta, ...
                tol, itermax, indfree)
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
            %       info = 5  Failure to converge within itermax iterations
            
            self.log.debug('-- Entering TRPCG --');
            self.log.debug(sprintf('tol = %7.3e, δ = %7.3e,', tol, delta));
            % Initialize the iterate w and the residual r.
            w = zeros(self.nlp.objSize, 1);
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
                self.log.debug(sprintf(['Leaving TRPCG, info', ...
                    '= %d (conv)'], info));
                return
            end
            
            for iters = 1:itermax
                appZero = max(self.eqTol * self.nlp.normJac * norm(p), ...
                    eps);
                Cp = self.nlp.fcon(p);
                chk = any(Cp(~indfree) >= appZero);
                if chk
                    % Project {p : (C*p)_i = 0} for i such as (C*x)_i = 0
                    p = self.eqProject(p, ~indfree);
                end
                
                %                 Cp = self.nlp.fcon(p); % Computes C*p
                %                 normCp = norm(Cp(~indfree)); % Should be near 0
                %                 self.log.debug(sprintf('\t||(C*p)_i|| = %7.3e', normCp));
                
                % Compute alpha and determine sigma such that the trust
                % region constraint || w + sigma*p || = delta is satisfied.
                q = Aprod(p);
                ptq = p'*q;
                self.log.debug(sprintf('\tp''*H*p = %7.3e', ptq));
                if ptq > 0
                    alpha = rho/ptq;
                else
                    alpha = 0;
                end
                
                sigma = Cflash.trqsol(w, p, delta);
                % Exit if there is negative curvature or if the
                % iterates exit the trust region.
                self.log.debug(sprintf('\tαCG = %7.1e, σ = %7.1e', ...
                    alpha, sigma));
                if (ptq <= 0 || alpha >= sigma)
                    if sigma ~= 0
                        w = w + sigma*p;
                    end
                    if ptq <= 0
                        info = 3;
                        self.log.debug(sprintf(['Leaving TRPCG, info', ...
                            ' = %d (negative curv)'], info));
                    else
                        info = 4;
                        self.log.debug(sprintf(['Leaving TRPCG, info', ...
                            ' = %d (exit TR)'], info));
                    end
                    return
                end
                
                % Update w and the residuals r.
                w = w + alpha*p;
                r = r - alpha*q;
                % Exit if the residual convergence test is satisfied.
                rtr = r'*r;
                rnorm = sqrt(rtr);
                self.log.debug(sprintf('\t||r''*r|| = %7.3e', rnorm));
                if rnorm <= tol
                    info = 1;
                    self.log.debug(sprintf(['Leaving TRPCG, info', ...
                        ' = %d (conv)'], info));
                    return
                end
                % Compute p = r + beta*p and update rho.
                beta = rtr/rho;
                p = r + beta*p;
                rho = rtr;
            end
            
            iters = itermax;
            info = 5;
            self.log.debug(sprintf('Leaving TRPCG, info = %d (fail)', ...
                info));
        end
        
        function [nbrpt, brptmin, brptmax] = breakpt(self, x, w)
            %% Breakpt
            % Find the breakpoints on the constraint set
            %                   cL <= C*x <= cU
            % from x in the direction w, i.e. finding alphas such as
            %                   C*(x + alpha*w) = cL
            % or
            %                   C*(x + alpha*w) = cU
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
            %             iLow = 1:self.nlp.objSize;
            %             iUpp = iLow;
            
            % Lower constraint intersections: Ax - cL > 0 & Aw < 0
            dec = (Cx(iLow) - self.nlp.cL(iLow) >= appZero) & Cw(iLow) < 0;
            % Upper constraint intersections: cU - Cx > 0 & Aw > 0
            inc = (self.nlp.cU(iUpp) - Cx(iUpp) >= appZero) & Cw(iUpp) > 0;
            
            nbrpt = sum(dec) + sum(inc);
            
            % Quick exit if no breakpoints
            if nbrpt == 0
                brptmin = 0;
                brptmax = 0;
                return
            end
            
            dec = iLow(dec);
            inc = iUpp(inc);
            
            brpt_dec = (self.nlp.cL(dec) - Cx(dec)) ./ Cw(dec);
            brpt_inc = (self.nlp.cU(inc) - Cx(inc)) ./ Cw(inc);
            
            % Finding min & max breakpoint
            brptmin =  inf;
            brptmax = -inf;
            
            if any(brpt_dec)
                brptmin_dec = min(brpt_dec);
                brptmin = min(brptmin, brptmin_dec);
                
                brptmax_dec = max(brpt_dec);
                brptmax = max(brptmax, brptmax_dec);
            end
            if any(brpt_inc)
                brptmin_inc = min(brpt_inc);
                brptmin = min(brptmin, brptmin_inc);
                
                brptmax_inc = max(brpt_inc);
                brptmax = max(brptmax, brptmax_inc);
            end
        end % function breakpt
        
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
            
            self.log.debug('-- Entering Armijo linesearch --');
            % Parsing optional arguments
            p = inputParser;
            p.addParameter('alpha0', 0.9999);
            p.addParameter('itermax', 20);
            p.addParameter('sTol', 1e-4);
            p.addParameter('stepTol', 1e-6);
            p.addParameter('tau', 0.5);
            p.parse(varargin{:});
            
            % Retrieving optional arguments
            alpha0 = p.Results.alpha0;
            % Interpolation factor for the linesearch
            tau = p.Results.tau;
            % Tolerance on the slope
            sTol = p.Results.sTol;
            % Lower bound on the value of the step size
            stepTol = p.Results.stepTol;
            % Maximal number of iterations for the backtracking procedure
            itermax = p.Results.itermax;
            
            % Only alpha values in ]0, 1[ are allowed
            assert((alpha0 < 1) & (alpha0 > 0));
            % Only tau values in ]0, 1[ are allowed
            assert((tau < 1) & (tau > 0));
            
            % Iteration counter
            iter = 1;
            
            % Getting objective function and gradient value at x
            [f, g] = self.nlp.obj(x);
            alph = alpha0;
            while true
                self.log.debug(sprintf('\tα = %7.1e', alph));
                % Armijo's condition
                if self.nlp.fobj(x + alph * s) > f + alph * sTol * g' * s
                    % Condition has not been met, reduce alpha
                    alph = alph * tau;
                else
                    % Return new value
                    x = x + alph * s;
                    info = true;
                    self.log.debug(sprintf(['Armijo linesearch', ...
                        ' is successful']));
                    return
                end
                if iter > itermax || alph < stepTol
                    % Too many iterations or step size too small
                    info = false;
                    self.log.debug(sprintf(['Failure of the', ...
                        ' Armijo linesearch']));
                    return
                end
                iter = iter + 1;
            end
            
        end % function armijoLineSearch
        
    end % methods (private)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    methods (Access = public, Hidden = true)
        
        function printf(self, varargin)
            fprintf(self.fid, varargin{:});
        end % function printf
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    methods (Access = private, Static)
               
        function xFree = Afree(xFree, Aprod, indfree, n)
            z = zeros(n,1);
            z(indfree) = xFree;
            z = Aprod(z);
            xFree = z(indfree);
        end % function Afree
        
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
        end % function trqsol
        
    end % methods(Static)
end