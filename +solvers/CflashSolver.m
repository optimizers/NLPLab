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
    %       * projectActiveFace
    %       * projectMixedSet
    %
    % A public attribute or method named
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
        maxIterCg; % maximum number of CG iterations per Newton step
        nSuccessIter; % number of successful iterations
        iterCg; % total number of CG iterations
        gNorm0; % norm of the gradient at x0
        mu0; % sufficient decrease parameter
        cgTol;
        fMin;
        fid; % File ID of where to direct log output
        
        % Backtracking parameters
        backtracking;
        suffDec;
        maxIterLS;
        
        maxExtraIter; % Limit number of extrapolations or interpolations
        
        eqTol; % Tolerance for equalities (see indFree)
        nProj; % # of projections (project + projectMixed)
        maxProj; % Maximal # of projections
        nEqProj; % # of projections on the active face
    end
    
    properties (Hidden = true, Constant)
        % Constants used to manipulate the TR radius. These are the numbers
        % used by TRON.
        sig1 = 0.25;
        sig2 = 0.50;
        sig3 = 4.00;
        eta0 = 1e-4;
        eta1 = 0.25;
        eta2 = 0.75;
        
        % Log header and body formats.
        LOG_HEADER_FORMAT = ['\n%5s  %13s  %13s  %9s  %9s  %5s  %9s', ...
            '  %9s  %6s  %6s %9s %9s\n'];
        LOG_BODY_FORMAT = ['%5d  %13.6e  %13.6e  %9d  %9d  %5d', ...
            '  %9.3e  %9.3e  %6s  %6d %9f %9.3e\n'];
        LOG_HEADER = {'iter', 'f(x)', '|g(x)|', '# Proj', '# EqProj', ...
            'cg', 'preRed', 'radius', 'status', '#free', 'time', 'precOpt'};
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
            elseif ~ismethod(nlp, 'projectMixed')
                error('No projectMixed method in nlp');
            elseif ~ismethod(nlp, 'projectActiveFace')
                error('No projectActiveFace method in nlp');
            elseif ~isprop(nlp, 'normJac')  && ~ismethod(nlp, 'normJac')
                error('No normJac attribute or method in nlp');
            end
            
            % Parse input parameters and initialize local variables.
            p = inputParser;
            p.KeepUnmatched = true;
            p.PartialMatching = false;
            p.addParameter('maxIterCg', length(nlp.x0));
            p.addParameter('cgTol', 0.1);
            p.addParameter('fMin', -1e32);
            p.addParameter('mu0', 0.01);
            p.addParameter('fid', 1);
            p.addParameter('maxProj', 1e5);
            p.addParameter('eqTol', 1e-12);
            p.addParameter('backtracking', false);
            p.addParameter('maxIterLS', 10);
            p.addParameter('suffDec', 1e-4);
            p.addParameter('maxExtraIter', 15);
            
            p.parse(varargin{:});
            
            self = self@solvers.NlpSolver(nlp, p.Unmatched);
            
            % Store various objects and parameters.
            self.cgTol = p.Results.cgTol;
            self.maxIterCg = p.Results.maxIterCg;
            self.fMin = p.Results.fMin;
            self.mu0 = p.Results.mu0;
            self.fid = p.Results.fid;
            self.eqTol = p.Results.eqTol;
            self.maxProj = p.Results.maxProj;
            self.backtracking = p.Results.backtracking;
            self.suffDec = p.Results.suffDec;
            self.maxIterLS = p.Results.maxIterLS;
            self.maxExtraIter = p.Results.maxExtraIter;
            
            if self.backtracking
                import linesearch.armijo;
            end
            
            import utils.PrintInfo;
            
        end % constructor
        
        function self = solve(self)
            %% Solve
            
            self.solveTime = tic;
            self.iter = 1;
            self.iterCg = 1;
            self.nSuccessIter = 0;
            self.iStop = self.EXIT_NONE;
            self.nProj = 0;
            self.nEqProj = 0;
            self.nlp.resetCounters();
            
            printObj = utils.PrintInfo('Cflash');
            
            % Make sure initial point is feasible
            x = self.project(self.nlp.x0);
            
            % First objective and gradient evaluation.
            [f, g] = self.nlp.obj(x);
            
            % Initialize stopping tolerance and initial TR radius.
            gNorm = norm(g);
            delta = gNorm;
            self.gNorm0 = gNorm;
            self.rOptTol = self.rOptTol * gNorm;
            self.rFeasTol = self.aFeasTol * abs(f);
            
            if self.verbose >= 2
                extra = containers.Map( ...
                    {'fMin', 'cgTol', 'mu0', 'maxProj', 'eqTol'}, ...
                    {self.fMin, self.cgTol, self.mu0, ...
                    self.maxProj, self.eqTol});
                printObj.header(self, extra);
                self.printf(self.LOG_HEADER_FORMAT, self.LOG_HEADER{:});
            end
            
            % Actual and predicted reductions. Initial inf value prevents
            % exits based on related on first iter.
            actRed = inf;
            preRed = inf;
            
            % Miscellaneous iter
            alphc = 1;
            status = '';
            
            %% Main loop
            while true
                
                % Check stopping conditions
                [~, nFree] = self.getIndFree(x);
                pgNorm = norm(self.gpstep(x, -1, g));
                now = toc(self.solveTime);
                
                % Optimality in the image space
                gPrec = real(self.nlp.prec.Inverse(g));
                mu = real(self.nlp.prec.Direct(x));
                precOpt = norm(max(0, mu - gPrec) - x);
                
                if pgNorm <= self.rOptTol + self.aOptTol
                    self.iStop = self.EXIT_OPT_TOL;
                elseif f < self.fMin
                    self.iStop = self.EXIT_UNBOUNDED;
                elseif (abs(actRed) <= (self.aFeasTol + self.rFeasTol)) ...
                        && (preRed  <= (self.aFeasTol + self.rFeasTol))
                    self.iStop = self.EXIT_FEAS_TOL;
                elseif self.iter >= self.maxIter
                    self.iStop = self.EXIT_MAX_ITER;
                elseif self.nlp.ncalls_fobj + self.nlp.ncalls_fcon >= ...
                        self.maxEval
                    self.iStop = self.EXIT_MAX_EVAL;
                elseif self.nProj >= self.maxProj
                    self.iStop = self.EXIT_MAX_PROJ;
                elseif now >= self.maxRT
                    self.iStop = self.EXIT_MAX_RT;
                end
                
                % Print current iter to log
                if self.verbose >= 2
                    self.printf(self.LOG_BODY_FORMAT, self.iter, f, ...
                        pgNorm, self.nProj, self.nEqProj, self.iterCg, ...
                        preRed, delta, status, nFree, now, precOpt);
                end
                
                % Act on exit conditions
                if self.iStop
                    self.x = x;
                    self.fx = f;
                    self.pgNorm = pgNorm;
                    break;
                end
                
                fc = f;
                xc = x;
                
                % Hessian operator
                H = self.nlp.hobj(x);
                
                % Cauchy step
                [alphc, s] = self.cauchy(H, x, g, delta, alphc);
                
                % Projected Newton step
                [x, s] = self.spcg(H, x, g, delta, s);
                
                % Compute the objective at this new point
                f = self.nlp.fobj(x);
                
                backtrackAttempted = false;
                while true
                    % Predicted reduction
                    Hs = self.nlp.hobjprod(x, [], s);
                    preRed = -(s' * g + 0.5 * s' * Hs);
                    % Actual reduction
                    actRed = fc - f;
                    % Update the trust-region radius
                    delta = self.updateDelta(f, fc, g, s, actRed, ...
                        preRed, delta);
                    
                    % Accept or reject step
                    if actRed > self.eta0 * preRed
                        % Successful step
                        status = '';
                        self.nSuccessIter = self.nSuccessIter + 1;
                        % Update the gradient value
                        g = self.nlp.gobj(x);
                        % Break loop if step is accepted
                        break;
                    elseif self.backtracking && ~backtrackAttempted
                        % The step is rejected, but we attempt to backtrack
                        [x, f] = linesearch.armijo(self, xc, fc, g, s);
                        backtrackAttempted = true;
                    else
                        % The step is rejected
                        status = 'rej';
                        % Fallback on previous values
                        f = fc;
                        x = xc;
                        % No backtracking is attempted or backtracking was
                        % already attempted. Exit loop.
                        break;
                    end
                end
                
                self.iter = self.iter + 1;
            end % main loop
            
            self.nObjFunc = self.nlp.ncalls_fobj + self.nlp.ncalls_fcon;
            self.nGrad = self.nlp.ncalls_gobj + self.nlp.ncalls_gcon;
            self.nHess = self.nlp.ncalls_hvp + self.nlp.ncalls_hes;
            
            %% End of solve
            self.solveTime = toc(self.solveTime);
            % Set solved attribute
            self.isSolved();
            
            printObj.footer(self);
        end % solve
        
        function xProj = project(self, x)
            %% Project - simple wrapper to increment nProj counter
            [xProj, solved] = self.nlp.project(x);
            if ~solved
                % Propagate throughout the program to exit
                self.iStop = self.EXIT_PROJ_FAILURE;
            end
            self.nProj = self.nProj + 1;
        end
        
        function xProj = projectMixed(self, x, fixed)
            %% ProjectMixed - simple wrapper to increment nProj counter
            [xProj, solved] = self.nlp.projectMixed(x, fixed);
            if ~solved
                % Propagate throughout the program to exit
                self.iStop = self.EXIT_PROJ_FAILURE;
            end
            self.nProj = self.nProj + 1;
        end
        
        function xProj = projectActiveFace(self, x, fixed)
            %% ProjectActiveFace - simple wrapper to increment nEqProj
            [xProj, solved] = self.nlp.projectActiveFace(x, fixed);
            if ~solved
                % Propagate throughout the program to exit
                self.iStop = self.EXIT_PROJ_FAILURE;
            end
            self.nEqProj = self.nEqProj + 1;
        end
        
    end % public methods
    
    
    methods (Access = private)
        
        function delta = updateDelta(self, f, fc, g, s, actRed, preRed, ...
                delta)
            %% UpdateDelta
            % Update trust-region radius according to a set of rules.
            
            snorm = norm(s);
            if self.nSuccessIter == 0
                delta = min(delta, snorm);
            end
            
            gts = g' * s;
            if f - fc - gts <= 0
                alph = self.sig3;
            else
                alph = max(self.sig1, -0.5 * gts / (f - fc - gts));
            end
            
            % Changing delta according to a set of rules:
            if actRed < self.eta0 * preRed || actRed == -inf
                delta = min(max(alph, self.sig1) * snorm, ...
                    self.sig2 * delta);
            elseif actRed < self.eta1 * preRed
                delta = max(self.sig1 * delta, min(alph * snorm, ...
                    self.sig2 * delta));
            elseif actRed < self.eta2 * preRed
                delta = max(self.sig1 * delta, min(alph * snorm, ...
                    self.sig3 * delta));
            else
                delta = max(delta, min(alph * snorm, self.sig3 * delta));
            end
        end
        
        function [alph, s] = cauchy(self, H, x, g, delta, alph)
            %% Cauchy
            % This subroutine computes a Cauchy step that satisfies a trust
            % region constraint and a sufficient decrease condition.
            %
            % The Cauchy step is computed for the quadratic
            %
            %       q(s) = 0.5*s'*H*s + g'*s,
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
            self.logger.debug(sprintf('α = %7.1e, δ = %7.3e', ...
                alph, delta));
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
                gts = g'*s;
                interp = (0.5 * s'*H*s + gts >= self.mu0 * gts);
            end
            
            % Either interpolate or extrapolate to find a successful step.
            if interp
                self.logger.debug('Interpolating');
                % Reduce alph until a successful step is found.
                while (toc(self.solveTime) < self.maxRT) && ~self.iStop
                    % This is a crude interpolation procedure that
                    % will be replaced in future versions of the code.
                    alph = interpf * alph;
                    s = self.gpstep(x, -alph, g);
                    sNorm = norm(s);
                    self.logger.debug(sprintf('\t||s|| = %7.3e', sNorm));
                    if sNorm <= delta
                        gts = g'*s;
                        if 0.5 * s'*H*s + gts < self.mu0 * gts
                            break
                        end
                    end
                end
            else % Extrapolate
                self.logger.debug('Extrapolating');
                % Increase alph until a successful step is found.
                alphas = alph;
                iter = 1;
                while (alph <= brptMax) && ...
                        (toc(self.solveTime) < self.maxRT) && ...
                        ~self.iStop && iter <= self.maxExtraIter
                    % This is a crude extrapolation procedure that
                    % will be replaced in future versions of the code.
                    alph = extrapf * alph;
                    s = self.gpstep(x, -alph, g);
                    sNorm = norm(s);
                    self.logger.debug(sprintf('\t||s|| = %7.3e', sNorm));
                    if sNorm <= delta
                        gts = g' * s;
                        q = 0.5 * s'*H*s + gts;
                        if q > self.mu0 * gts
                            break
                        end
                        alphas = alph;
                    else
                        break
                    end
                    iter = iter + 1;
                end
                % Recover the last successful step.
                alph = alphas;
                s = self.gpstep(x, -alph, g);
            end
            self.logger.debug(sprintf('Leaving Cauchy, α = %7.1e', alph));
        end % cauchy
        
        function s = prsrch(self, H, x, g, w, fixed)
            %% PrSrch - Projected search
            % s = prsrch(H, x, g, w, fixed) where
            % Inputs:
            %     H is an opSpot to compute matrix-vector
            %     products
            %     x        current point
            %     g        current gradient
            %     w        search direction
            % Output:
            %     s is the step s[alph]
            %
            %   This subroutine uses a projected search to compute a step
            %   that satisfies a sufficient decrease condition for the
            %   quadratic
            %
            %           q(s) = 0.5*s'*H*s + g'*s,
            %
            %   where H is a symmetric matrix in compressed column
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
            
            % Find the smallest break-point on the ray x + alph*w.
            [~, brptMin, ~] = self.breakpt(x, w);
            self.logger.debug(sprintf('brptMin = %7.1e', brptMin));
            
            % Reduce alph until the sufficient decrease condition is
            % satisfied or x + alph*w is feasible.
            self.logger.debug('Interpolating');
            iter = 1;
            while (alph > brptMin) && ...
                    (toc(self.solveTime) < self.maxRT) && ~self.iStop ...
                    && iter <= self.maxExtraIter
                
                % Calculate P[x + alph*w] - x and check the sufficient
                % decrease condition.
                s = self.projectMixed(x + alph * w, fixed) - x;
                self.logger.debug(sprintf('\t||s|| = %7.3e', norm(s)));
                gts = g' * s;
                if 0.5 * s'*H*s + gts <= self.mu0 * gts
                    return
                end
                % This is a crude interpolation procedure that
                % will be replaced in future versions of the code.
                alph = interpf * alph;
                iter = iter + 1;
            end
            % Force at least one more constraint to be added to the active
            % set if alph < brptMin and the full step is not successful.
            % There is sufficient decrease because the quadratic function
            % is decreasing in the ray x + alph*w for 0 <= alph <= 1.
            if alph < 1 && alph < brptMin
                alph = brptMin;
            end
            
            % Compute the final iterate and step.
            s = self.projectMixed(x + alph * w, fixed) - x;
        end % prsrch
        
        function [x, s] = spcg(self, H, x, g, delta, s)
            %% SPCG - Minimize a linearly constrained quadratic.
            %
            % This subroutine generates a sequence of approximate
            % minimizers for the subproblem
            %
            %       min { q(x) : cL <= C*x <= cU }.
            %
            % The quadratic is defined by
            %
            %       q(x[0]+s) = 0.5*s'*H*s + g'*s,
            %
            % where x[0] is a base point provided by the user, H is an
            % opSpot of the hessian, and g is a
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
            %       || (g + H*s)[free] || <= rtol*|| g[free] ||
            %
            % In this case the final x is an approximate minimizer in the
            % face defined by the free variables.
            %
            % The subroutine terminates when the trust region bound does
            % not allow further progress, that is, || p[k] || = delta.
            % In this case the final x satisfies q(x) < q(x[k]).
            
            self.logger.debug('-- Entering SPCG --');
            % Compute the Cauchy point. This point should already respect
            % the constraint set.
            x = x + s;
            % Evaluate H * (x_{k, j} - x_k)
            Hs = H * s;
            
            % There are at most n iterations because at each iter at least
            % one variable becomes active.
            iters = 1;
            for nFaces = 1 : self.nlp.n
                
                % Determine the free variables at the current minimizer.
                [fixed, nFree] = self.getIndFree(x);
                % getIndFree returns the free indices, fixed = ~ free
                fixed = ~fixed;
                % Exit if there are no free constraints
                if nFree == 0
                    break;
                end
                
                % Hs & g must be such that (Cp)_i = 0 \forall i \in fixed
                Hs = self.projectActiveFace(Hs, fixed);
                gP = self.projectActiveFace(g, fixed);
                % Compute the grad of quad q = g + H*(x{k, j} - x_k)
                gQuad = Hs + gP;
                
                % Norm of obj. func. gradient is a stopping criteria
                gNorm = norm(gP);
                
                % Solve the trust region subproblem in the free variables
                % to generate a direction p[k]. Store p[k] in the array w.
                tol = self.cgTol * gNorm;
                [w, iterTR, infoTR] = self.trpcg(H, gQuad, delta, ...
                    tol, self.maxIterCg, fixed, s);
                iters = iters + iterTR;
                
                % Use a projected search to obtain the next iterate.
                % The projected search algorithm stores s[k] in w.
                w = self.prsrch(H, x, gQuad, w, fixed);
                
                % Update the minimizer and the step.
                % Note that s now contains x[k+1] - x[0].
                x = x + w;
                s = s + w;
                
                % Compute H*(x[k+1] - x[0]) and store in w.
                Hs = H * s;
                HsP = self.projectActiveFace(Hs, fixed);
                
                % Convergence and termination test.
                % We terminate if the preconditioned conjugate gradient
                % method encounters a direction of negative curvature, or
                % if the step is at the trust region bound.
                if norm(HsP + gP) <= tol || infoTR == 2 || infoTR == 3 ...
                        || iters > self.maxIterCg || ...
                        toc(self.solveTime) >= self.maxRT || self.iStop
                    self.logger.debug('Leaving SPCG');
                    break;
                end
            end % faces
            self.iterCg = self.iterCg + iters;
        end % spcg
        
        function s = gpstep(self, x, alph, w)
            %% GpStep - Compute the gradient projection step.
            % s = P[x + alph*w] - x,
            % where P is the projection onto the linear constraint set
            s = self.project(x + alph*w) - x;
        end
        
        function [indFree, nFree] = getIndFree(self, x)
            %% GetIndFree - Find indices of the free variables
            % Assuming linear constraints
            % cL <= C*x <= cU
            % where C is the jacobian of the linear constraints
            Cx = self.nlp.fcon(x); % C*x
            % Represents "relative" zero value, smallest approx is eps
            appZero = self.getAppZero(x);
            % The equalities cU - C*x = 0 and C*x - cL = 0 are not likely
            % to happen in practice
            indFree = (self.nlp.cU - Cx >= appZero) & ...
                (Cx - self.nlp.cL >= appZero);
            if nargout > 1
                nFree = sum(indFree);
            end
        end
        
        function [w, iters, info] = trpcg(self, H, g, delta, ...
                tol, iterMax, fixed, s)
            %% TRPCG
            % This subroutine uses a truncated conjugate gradient method to
            % find an approximate minimizer of the trust-region subproblem
            %
            %       min_s   q(s)
            %       sc      || s || <= delta
            %               c_i'*s = 0,         for i in A(x):= c_i'*x = 0
            %
            % where q is the quadratic
            %
            %       q(s) = 0.5*s'*H*s + g'*s,
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
            %       info = 2  Negative curvature direction generated.
            %                 In this case || w || = delta.
            %
            %       info = 3  Conjugate gradient iterates exit the
            %                 trust region. In this case || w || = delta.
            %
            %       info = 4  Failure to converge within iterMax iterations
            
            self.logger.debug('-- Entering TRPCG --');
            self.logger.debug(sprintf( ...
                'tol = %7.3e, δ = %7.3e,', tol, delta));
            % Initialize the iterate w and the residual r.
            w = zeros(self.nlp.n, 1);
            % Initialize the residual r of grad q to -g.
            r = -g; % g is already projected
            %             r = self.projectActiveFace(-g, fixed);
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
                
                % Compute alph and determine sigma such that the trust
                % region constraint || w + sigma*p || = delta is satisfied.
                q = H * p;
                ptq = p'*q;
                self.logger.debug(sprintf('\tp''*H*p = %7.3e', ptq));
                if ptq > 0
                    alph = rho/ptq;
                else
                    alph = 0;
                end
                
                sigma = solvers.CflashSolver.trqsol(w + s, p, delta);
                % Exit if there is negative curvature or if the
                % iterates exit the trust region.
                self.logger.debug(sprintf('\tαCG = %7.1e, σ = %7.1e', ...
                    alph, sigma));
                if (ptq <= 0 || alph >= sigma)
                    if sigma ~= 0
                        w = w + sigma*p;
                    end
                    if ptq <= 0
                        info = 2;
                        self.logger.debug(sprintf( ...
                            ['Leaving TRPCG, info', ...
                            ' = %d (negative curv)'], info));
                    else
                        info = 3;
                        self.logger.debug(sprintf( ...
                            ['Leaving TRPCG, info', ...
                            ' = %d (exit TR)'], info));
                    end
                    return
                end
                
                % Update w and the residuals r.
                w = w + alph*p;
                r = r - alph*self.projectActiveFace(q, fixed);
                % Exit if the residual convergence test is satisfied.
                rtr = r'*r;
                rnorm = sqrt(rtr);
                self.logger.debug(sprintf('\t||r''*r|| = %7.3e', rnorm));
                if rnorm <= tol
                    info = 1;
                    self.logger.debug(sprintf(['Leaving TRPCG, info', ...
                        ' = %d (conv)'], info));
                    return
                elseif toc(self.solveTime) >= self.maxRT
                    break
                end
                % Compute p = r + betaFactor*p and update rho.
                betaFactor = rtr/rho;
                % Since r is always projected, p is also projected
                p = r + betaFactor*p;
                rho = rtr;
            end % for loop
            
            info = 4;
            self.logger.debug(sprintf( ...
                'Leaving TRPCG, info = %d (fail)', info));
        end % trpcg
        
        function [nBrpt, brptMin, brptMax] = breakpt(self, x, w)
            %% BreakPt
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
            appZero = self.getAppZero(x);
            
            % Lower constraint intersections: C*x - cL > 0 & C*w < 0
            dec = (Cx - self.nlp.cL >= appZero) & (Cw <= appZero);
            % Upper constraint intersections: cU - C*x > 0 & C*w > 0
            inc = (self.nlp.cU - Cx >= appZero) & (Cw >= appZero);
            
            dec = dec & self.nlp.iLow;
            inc = inc & self.nlp.iUpp;
            
            nBrpt = sum(dec | inc);
            
            % Quick exit if no breakpoints
            if nBrpt == 0
                brptMin = 0;
                brptMax = 0;
                return
            end
            
            brptDec = (self.nlp.cL(dec) - Cx(dec)) ./ Cw(dec);
            brptInc = (self.nlp.cU(inc) - Cx(inc)) ./ Cw(inc);
            
            brptMin = min([brptInc; brptDec]);
            brptMax = max([brptInc; brptDec]);
        end % breakpt
        
        function appZero = getAppZero(self, x)
            appZero = max(self.eqTol * self.nlp.normJac * norm(x), eps);
        end
        
        
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
