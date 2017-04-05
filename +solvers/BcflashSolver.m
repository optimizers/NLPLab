classdef BcflashSolver < solvers.NlpSolver
    %% BcflashSolver
    
    
    properties (SetAccess = private, Hidden = false)
        maxIterCg; % maximum number of CG iters per Newton step
        nSuccessIter; % number of successful iters
        iterCg; % total number of CG iters
        gNorm0; % norm of the gradient at x0
        mu0; % sufficient decrease parameter
        cgTol;
        fMin;
        fid; % File ID of where to direct log output
        
        % Initialize cauchy with Barzilai-Borwein step length
        useBb;
        % Backtracking parameters
        backtracking;
        suffDec;
        maxIterLS;
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
        LOG_HEADER_FORMAT = '\n%5s  %13s  %13s  %5s  %9s  %9s  %6s  %9s\n';
        LOG_BODY_FORMAT = ['%5i  %13.6e  %13.6e  %5i  %9.3e  %9.3e', ...
            '  %6s  %9d\n'];
        LOG_HEADER = {'iter', 'f(x)', '|g(x)|', 'cg', 'preRed', ...
            'radius', 'status', 'nFree'};
    end % constant properties
    
    
    methods (Access = public)
        
        function self = BcflashSolver(nlp, varargin)
            %% Constructor
            
            % Parse input parameters and initialize local variables
            p = inputParser;
            p.KeepUnmatched = true;
            p.PartialMatching = false;
            p.addParameter('maxIterCg', length(nlp.x0));
            p.addParameter('cgTol', 0.1);
            p.addParameter('fMin', -1e32);
            p.addParameter('mu0', 0.01);
            p.addParameter('fid', 1);
            p.addParameter('useBb', true);
            p.addParameter('backtracking', true);
            p.addParameter('maxIterLS', 10);
            p.addParameter('suffDec', 1e-4);

            p.parse(varargin{:});
            
            self = self@solvers.NlpSolver(nlp, p.Unmatched);
            
            % Store various objects and parameters
            self.cgTol = p.Results.cgTol;
            self.maxIterCg = p.Results.maxIterCg;
            self.fMin = p.Results.fMin;
            self.mu0 = p.Results.mu0;
            self.fid = p.Results.fid;
            self.useBb = p.Results.useBb;
            self.backtracking = p.Results.backtracking;
            self.suffDec = p.Results.suffDec;
            self.maxIterLS = p.Results.maxIterLS;
            
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
            self.iStop = self.EXIT_NONE;
            self.nlp.resetCounters();
            
            
            printObj = utils.PrintInfo('bcflash');
            
            if self.verbose >= 2
                extra = containers.Map({'fMin', 'cgTol', 'mu0'}, ...
                    {self.fMin, self.cgTol, self.mu0});
                printObj.header(self, extra);
                self.printf(self.LOG_HEADER_FORMAT, self.LOG_HEADER{:});
            end
            
            % Make sure initial point is feasible
            x = self.project(self.nlp.x0);
            
            % First objective and gradient evaluation.
            [f, g] = self.nlp.obj(x);
            
            % Initialize stopping tolerance and initial TR radius
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
            status = '';
            
            %% Main loop
            while ~self.iStop
                % Check stopping conditions
                pgNorm = norm(self.gpstep(x, -1, g));
                if pgNorm <= self.rOptTol + self.aOptTol
                    self.iStop = self.EXIT_OPT_TOL;
                elseif f < self.fMin
                    self.iStop = self.EXIT_UNBOUNDED;
                elseif (abs(actRed) <= (self.aFeasTol + self.rFeasTol)) ...
                        && (preRed  <= (self.aFeasTol + self.rFeasTol))
                    self.iStop = self.EXIT_FEAS_TOL;
                elseif self.iter >= self.maxIter
                    self.iStop = self.EXIT_MAX_ITER;
                elseif toc(self.solveTime) >= self.maxRT
                    self.iStop = self.EXIT_MAX_RT;
                end
                
                % Print current iter to log
                if self.verbose >= 2
                    [~, nFree] = self.getIndFree(x);
                    self.printf(self.LOG_BODY_FORMAT, self.iter, f, ...
                        pgNorm, self.iterCg, preRed, delta, status, nFree);
                end
                
                % Act on exit conditions
                if self.iStop
                    self.x = x;
                    self.fx = f;
                    self.pgNorm = pgNorm;
                    break
                end
                
                fc = f;
                xc = x;
                
                % Hessian operator
                H = self.nlp.hobj(x);
                
                % Cauchy step
                [alphc, s] = self.cauchy(H, x, g, delta, alphc);
                
                % Projected Newton step
                [x, s, ~] = self.spcg(H, x, g, delta, s);
                
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
                    if actRed > self.eta0 * preRed;
                        % Successful step
                        status = '';
                        self.nSuccessIter = self.nSuccessIter + 1;
                        if self.useBb % Init cauchy with BB step length
                            % Can only use BB step if iteration successful
                            gOld = g;
                            % Update the gradient value
                            g = self.nlp.gobj(x);
                            % Update initial alphc used in Cauchy
                            alphc = self.bbStepLength(xc, x, gOld, g);
                        else
                            % Update the gradient value
                            g = self.nlp.gobj(x);
                        end
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
        
        function x = project(self, x, ind)
            %% Project
            % Project a vector onto the box defined by bL, bU.
            if nargin > 2
                x = min(self.nlp.bU(ind), max(x, self.nlp.bL(ind)));
            else
                x = min(self.nlp.bU, max(x, self.nlp.bL));
            end
        end
        
    end % public methods
    
    
    methods (Access = private)
        
        function alph = bbStepLength(self, xOld, x, gOld, g)
            %% BBStepLength - Compute Barzilai-Borwein step length
            % alph_BB = (s' * s) / (s' * y)
            s = x - xOld;
            % Denominator of Barzilai-Borwein step length
            betaBB = s' * (g - gOld);
            
            % Find the minimal and maximal break-point on x - alph*g.
            [~, alphMin, alphMax] = self.breakpt(x, -g);
            % Safeguard
            if isinf(alphMax)
                alphMax = 1e3;
            end
            if isinf(alphMin)
                alphMin = 1e-3;
            end
            
            if betaBB < 0
                % Fall back to maximal step length
                alph = alphMax;
            else
                % Compute Barzilai-Borwein step length
                % y = g - gOld
                % alph_BB = (s' * s) / (s' * y)
                % Assert alph \in [alph_min, alph_max]
                alph = min(alphMax, ...
                    max(alphMin, (s' * s) / betaBB));
            end
        end % bbsteplength
        
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
            if actRed < self.eta0 * preRed || actRed == -inf;
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
            % with P the projection onto the n-dimensional interval
            % [bL, bU]. The Cauchy step satisfies the trust region
            % constraint and the sufficient decrease condition
            %
            %       || s || <= delta,      q(s) <= mu_0*(g'*s),
            %
            % where mu_0 is a constant in (0,1).
            
            interpf =  0.1;     % interpolation factor
            extrapf = 1 / interpf;     % extrapolation factor
            
            % Find the minimal and maximal break-point on x - alph*g.
            [~, ~, brptMax] = self.breakpt(x, -g);
            
            % Evaluate the initial alph and decide if the algorithm
            % must interpolate or extrapolate.
            s = self.gpstep(x, -alph, g);
            
            if norm(s) >= delta
                interp = true;
            else
                gts = g'*s;
                interp = (0.5*s'*H*s + gts >= self.mu0*gts);
            end
            
            % Either interpolate or extrapolate to find a successful step.
            if interp
                % Reduce alph until a successful step is found.
                while (toc(self.solveTime) < self.maxRT)
                    % This is a crude interpolation procedure that
                    % will be replaced in future versions of the code.
                    alph = interpf * alph;
                    s = self.gpstep(x, -alph, g);
                    if norm(s) <= delta
                        gts = g'*s;
                        if 0.5 * s'*H*s + gts < self.mu0 * gts
                            break
                        end
                    end
                end
            else
                % Increase alph until a successful step is found.
                alphs = alph;
                while (alph <= brptMax) && ...
                        (toc(self.solveTime) < self.maxRT)
                    % This is a crude extrapolation procedure that
                    % will be replaced in future versions of the code.
                    alph = extrapf * alph;
                    s = self.gpstep(x, -alph, g);
                    if norm(s) <= delta
                        gts = g' * s;
                        if 0.5 * s'*H*s + gts > self.mu0 * gts
                            break
                        end
                        alphs = alph;
                    else
                        break
                    end
                end
                % Recover the last successful step.
                alph = alphs;
                s = self.gpstep(x, -alph, g);
            end
            
        end % cauchy
        
        function s = prsrch(self, H, x, g, w, indFree)
            %% PRSRCH - Projected search.
            % s = prsrch(H, x, g, w) where
            % Inputs:
            %     H is an opSpot to compute matrix-vector
            %     products
            %     x        current point with respect to indFree
            %     g        current gradient with respect to indFree
            %     w        search direction with respect to indFree
            % Output:
            %     s (the new w) is the step with respect to indFree
            %
            % !!! ALL INPUT/OUTPUT VARIABLES ARE OF REDUCED SIZE !!!
            %
            %     This subroutine uses a projected search to compute a step
            %     that satisfies a sufficient decrease condition for the
            %     quadratic
            %
            %           q(s) = 0.5*s'*H*s + g'*s,
            %
            %     where A is a symmetric matrix in compressed column
            %     storage, and g is a vector. Given the parameter alph,
            %     the step is
            %
            %           s[alph] = P[x + alph*w] - x,
            %
            %     where w is the search direction and P the projection onto
            %     the n-dimensional interval [bL,bU]. The final step
            %     s = s[alph] satisfies the sufficient decrease condition
            %
            %           q(s) <= mu_0*(g'*s),
            %
            %     where mu_0 is a constant in (0,1).
            %
            %     The search direction w must be a descent direction for
            %     the quadratic q at x such that the quadratic is
            %     decreasing in the ray  x + alph*w for 0 <= alph <= 1.
            
            interpf = 0.5; % Interpolation factor
            
            % Set the initial alph = 1 because the quadratic function is
            % decreasing in the ray x + alph*w for 0 <= alph <= 1.
            alph = 1;
            
            % Find the smallest break-point on the ray x + alph*w.
            [~, brptMin, ~] = self.breakpt(x, w, indFree);
            
            % Reduce alph until the sufficient decrease condition is
            % satisfied or x + alph*w is feasible.
            while (alph > brptMin) && (toc(self.solveTime) < self.maxRT)
                % Calculate P[x + alph*w] - x and check the sufficient
                % decrease condition.
                s = self.gpstep(x, alph, w, indFree);
                gts = g' * s;
                if 0.5 * s'*H*s + gts <= self.mu0 * gts
                    break;
                end
                % This is a crude interpolation procedure that
                % will be replaced in future versions of the code.
                alph = interpf * alph;
            end
            
            % Force at least one more constraint to be added to the active
            % set if alph < brptMin and the full step is not successful.
            % There is sufficient decrease because the quadratic function
            % is decreasing in the ray x + alph*w for 0 <= alph <= 1.
            if alph < 1 && alph < brptMin
                alph = brptMin;
            end
            % Compute the final iterate and step.
            s = self.gpstep(x, alph, w, indFree); % s = P[x + alph*w] - x
        end % prsrch
        
        function [x, s, info] = spcg(self, H, xk, g, delta, s)
            %% SPCG - Minimize a bound-constraint quadratic
            %
            % This subroutine generates a sequence of approximate
            % minimizers for the subproblem
            %
            %       min { q(x) : bL <= x <= bU }.
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
            %       min { q(x[k]+p) : || L'*p || <= delta, s(fixed) = 0 },
            %
            % where fixed is the set of variables fixed at x[k] and delta is
            % the trust region bound.
            %
            % Given p[k], the next minimizer x[k+1] is generated by a
            % projected search.
            %
            % The starting point for this subroutine is x[1] = x[0] + s,
            % where x[0] is a base point and s is the Cauchy step.
            %
            % The subroutine converges when the step s satisfies
            %
            %       || (g + H*s)[free] || <= rTol*|| g[free] ||
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
            %                || (g + H*s)[free] || <= rTol*|| g[free] ||,
            %                and the final x is an approximate minimizer
            %                in the face defined by the free variables.
            %
            %      info = 2  Termination. The trust region bound does
            %                not allow further progress.
            %
            %      info = 3  Failure to converge within iterMax iters.
            Hs = H * s;
            % Compute the Cauchy point
            x = xk + s;
            % Start the main iter loop.
            % There are at most n iters because at each iter
            % at least one variable becomes active.
            iters = 1;
            for nfaces = 1:self.nlp.n
                
                % Determine the free variables at the current minimizer.
                % The indices of the free variables are stored in the first
                % n free positions of the array indFree.
                indFree = self.getIndFree(x);
                
                % Exit if there are no free constraints
                if ~any(indFree)
                    info = 1;
                    break;
                end
                
                % Compute the gradient grad q(x[k]) = g + H*(x[k] - x[0]),
                % of q at x[k] for the free variables.
                % Recall that w contains  H*(x[k] - x[0]).
                % Compute the norm of the reduced gradient Z'*g.
                wa = g(indFree);
                gFree = Hs(indFree) + wa;
                gfNorm = norm(wa);
                
                % Solve the trust region subproblem in the free variables
                % to generate a direction p[k]. Store p[k] in the array w.
                tol = self.cgTol * gfNorm;
                
                Hfree = H(indFree, indFree);
                
                [w, iterTR, infoTR] = solvers.BcflashSolver.trpcg( ...
                    Hfree, gFree, delta, tol, self.maxIterCg, s(indFree));
                iters = iters + iterTR;
                
                % Use a projected search to obtain the next iterate.
                % The projected search algorithm stores s[k] in w.
                w = self.prsrch(Hfree, x(indFree), gFree, w, ...
                    indFree);
                
                % Update the minimizer and the step.
                % Note that s now contains x[k+1] - x[0].
                x(indFree) = x(indFree) + w;
                s(indFree) = s(indFree) + w;
                
                % Compute A*(x[k+1] - x[0]) and store in w.
                Hs = H * s;
                
                % Convergence and termination test.
                % We terminate if the preconditioned conjugate gradient
                % method encounters a direction of negative curvature, or
                % if the step is at the trust region bound.
                if norm(Hs(indFree) + g(indFree)) <= self.cgTol * gfNorm
                    info = 1;
                    break;
                elseif infoTR == 2 || infoTR == 3
                    info = 2;
                    break;
                elseif iters > self.maxIterCg
                    info = 3;
                    break;
                end
                
            end % faces
            
            self.iterCg = self.iterCg + iters;
        end % spcg
        
        function [indFree, nFree] = getIndFree(self, x)
            %% GetIndFree
            % Find the free variables
            indFree = (self.nlp.bL < x) & (x < self.nlp.bU);
            if nargout > 1
                nFree = sum(indFree);
            end
        end
        
        function s = gpstep(self, x, alph, w, ind)
            %% GpStep - Compute the gradient projection step.
            % Compute the gradient projection step
            %
            % s = P[x + alph*w] - x,
            %
            % where P is the projection onto the box defined by bL, bU.
            if nargin > 4
                s = self.project(x + alph * w, ind) - x;
            else
                s = self.project(x + alph * w) - x;
            end
        end
        
        function [nBrpt, brptMin, brptMax] = breakpt(self, x, w, ind)
            %% BreakPt
            % This subroutine computes the number of break-points, and
            % the minimal and maximal break-points of the projection of
            % x + w on the n-dimensional interval [bL,bU].
            
            if nargin > 3
                bU = self.nlp.bU(ind);
                bL = self.nlp.bL(ind);
            else
                bU = self.nlp.bU;
                bL = self.nlp.bL;
            end
            
            inc = x < bU & w > 0;     % upper bound intersections
            dec = x > bL & w < 0;     % lower bound intersections
            
            nBrpt = sum(inc | dec);   % number of breakpoints
            
            % Quick exit if no breakpoints
            if nBrpt == 0
                brptMin = 0;
                brptMax = 0;
                return
            end
            
            brptInc = (bU(inc) - x(inc)) ./ w(inc);
            brptDec = (bL(dec) - x(dec)) ./ w(dec);
            
            brptMin = min([brptInc; brptDec; inf]);
            brptMax = max([brptInc; brptDec; -inf]);
        end % breakpt
        
    end % private methods
    
    
    methods (Access = public, Hidden = true)
        
        function printf(self, varargin)
            fprintf(self.fid, varargin{:});
        end
        
    end % hidden public methods
    
    
    methods (Access = private, Static)
        
        function [w, iters, info] = trpcg(H, g, delta, tol, iterMax, s)
            %% TRPCG - Trust-region projected conjugate gradient
            % This subroutine uses a truncated conjugate gradient method to
            % find an approximate minimizer of the trust-region subproblem
            %
            %       min { q(s) : ||s|| <= delta }.
            %
            % where q is the quadratic
            %
            %       q(s) = 0.5*s'*H*s + g'*s,
            %
            % H is an opSpot of the reduced hessian and g is a vector.
            %
            % Termination occurs if the conjugate gradient iterates leave
            % the trust region, a negative curvature direction is
            % generated, or one of the following two convergence tests is
            % satisfied.
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
            %       info = 4  Failure to converge within iterMax iters.
            
            n = length(g);
            % Initialize the iterate w and the residual r.
            w = zeros(n, 1);
            % Initialize the residual r of grad q to -g.
            r = -g;
            % Initialize the direction p.
            p = r;
            % Initialize rho and the norms of r and t.
            rho = r'*r;
            rnorm0 = sqrt(rho);
            
            % Exit if g = 0.
            if rnorm0 == 0
                iters = 0;
                info = 1;
                return
            end
            
            for iters = 1:iterMax
                Hp = H * p;
                q = Hp;
                % Compute alph and determine sigma such that the TR
                % constraint || w + sigma*p || = delta is satisfied.
                ptq = p'*q;
                if ptq > 0
                    alph = rho/ptq;
                else
                    alph = 0;
                end
                sigma = solvers.BcflashSolver.trqsol(w + s, p, delta);
                % Exit if there is negative curvature or if the
                % iterates exit the trust region.
                if (ptq <= 0 || alph >= sigma)
                    if sigma ~= 0
                        w = w + sigma*p;
                    end
                    if ptq <= 0
                        info = 2;
                    else
                        info = 3;
                    end
                    return
                end
                % Update w and the residuals r.
                w = w + alph * p;
                r = r - alph * q;
                
                % Exit if the residual convergence test is satisfied.
                rtr = r'*r;
                rnorm = sqrt(rtr);
                if rnorm <= tol
                    info = 1;
                    return
                end
                
                % Compute p = r + betaFact*p and update rho.
                betaFact = rtr/rho;
                p = r + betaFact*p;
                rho = rtr;
            end % for loop
            
            info = 4;
        end % trpcg
        
        function sigma = trqsol(x, p, delta)
            %% TRQSOL - Largest solution of the TR equation.
            %     This subroutine computes the largest (non-negative)
            %     solution of the quadratic trust region equation
            %
            %           ||x + sigma*p|| = delta.
            %
            %     The code is only guaranteed to produce a non-negative
            %     solution if ||x|| <= delta, and p != 0. If the trust
            %     region equation has no solution, sigma = 0.
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