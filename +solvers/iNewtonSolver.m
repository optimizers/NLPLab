classdef iNewtonSolver < solvers.UnconstrainedSolver
  %
  % This code comes with no guarantee or warranty of any kind.
  %
  % [x, histout, costdata] = inewton(nlp, parms)
  %
  % Inexact Newton method for unconstrained optimization.
  %
  % Input: nlp = NLPModel instance (constraints will not be taken into account)
  %
  %        parms.atol      = absolute stopping tolerance   (1.0e-8)
  %        parms.rtol      = relative stopping tolerance   (1.0e-6)
  %        parms.maxit     = maximum number of iterations  (max(2*n, 150))
  %        parms.mem       = LBFGS memory parameter        (5)
  %
  % The method stops as soon as ‖∇f(x)‖ ≤ atol + rtol * ‖∇f(x0)‖.
  %
  % Output: x = solution
  %         histout = iteration history
  %             Each row of histout is
  %              [norm(grad), f, shift, inner iteration count]
  %         costdata = [num f, num grad, num hess]
  %

  properties (SetAccess=private)
    hdr_fmt = '%4s  %7s  %9s  %5s  %7s  %7s  %s';
    fmt = '%7.1e  %9.2e  %5d  %7.1e  %7.1e  ';

    % Armijo linesearch parameters.
    slope_factor = 1.0e-4;
    bktrk_factor = 1.5;
    bk_max = 5;
  end

  methods

    % Constructor
    function self = iNewtonSolver(nlp, varargin)
      self = self@solvers.UnconstrainedSolver(nlp, varargin{:});
    end

    function self = solve(self)

      % Default parameters.
      nlp = self.nlp;
      n = nlp.n;
      x = nlp.x0;
      y = [];  % Dummy Lagrange multipliers for hlagprod.
      iter_history = zeros(self.max_iter, 5);

      % Initialize.
      sub_opts.rtol = 1.0e-3;
      sub_opts.show = false;  % true;
      status_msg = 'unknown';

      [f, g] = nlp.obj(x);
      %H = opFunction(n, n, @(v, mode) nlp.hlagprod(x, y, v));
      H = nlp.hobj(x);

      gNorm = norm(g);
      self.gNorm0 = gNorm;
      tol = self.opt_atol + self.opt_rtol * gNorm;
      iter_history(self.iter + 1, :) = [gNorm, f, 0, 0, 0];

      hdr = sprintf(self.hdr_fmt, ...
                    'Iter', '‖∇f(x)‖', 'f(x)', 'Inner', 'shift', 'step', 'er');
      self.logger.info(hdr);
      log_line = [sprintf('%4d  ', self.iter), ...
                  sprintf(self.fmt, iter_history(self.iter + 1, :))];
      self.logger.info(log_line);

      while gNorm > tol & self.iter <= self.max_iter

        self.iter = self.iter + 1;

        %descent = false;
        %sub_opts.shift = 0;
        %flags.niters = 0;
        %while ~descent
          %d = (H + sub_opts.shift * speye(n)) \ g;
          %gd = g' * d;
          %descent = (gd > 1.0e-6 * gNorm * norm(d));
          %shift_old = sub_opts.shift;
          %sub_opts.shift = sub_opts.shift + norm(H,1);
          %flags.niters = flags.niters + 1;
        %end
        %stats.istop = 1;
        %sub_opts.shift = shift_old;  % Increased one too many times.

        % Compute search direction.
        sub_opts.shift = 0;
        Anorm = 0;
        Acond = 1;
        sub_opts.rtol = max(1.0e-10, min(min(0.01, sqrt(gNorm)) * gNorm, 1.0e-4));
        descent = false;
        while ~descent
          % Estimate of the smallest eigenvalue of A using |A|/cond(A).
          % Recall that |A|_2 ≤ |A|_F ≤ √n |A|_2, so that
          % ‖A‖_F / condF(A) / √n ≤ λmin(A) = ‖A‖_2 / cond2(A) ≤ ‖A‖_F / condF(A).
          eigmin = Anorm / Acond / sqrt(n);
          %eigmin = Anorm / Acond;
          %sub_opts.shift = sub_opts.shift - Anorm / 100; % frob(A) ≥ norm(A).
          sub_opts.shift = sub_opts.shift - eigmin / 10;

          %sub_opts.rtol = sub_opts.rtol / 10;
          sub_opts.itnlim = n;
          [d, flags, stats] = minres_spot(H, g, sub_opts);
          Anorm = stats.Anorm;
          condA = stats.Acond;

          gd = g' * d;
          dNorm = norm(d);
          descent = (gd > 1.0e-4 * gNorm * dNorm) & flags.solved;  % -d is a descent dir.
        end
        if stats.istop > 4 & stats.istop < 10
          d = g;
        end

        % Perform line search
        nbk = 0;
        step = 1;
        xt = x - step * d;
        ft = nlp.obj(xt);
        while (nbk < self.bk_max) & (ft > f - self.slope_factor * step * gd)
          nbk = nbk + 1;
          step = step / self.bktrk_factor;
          xt = x - step * d;
          ft = nlp.obj(xt);
        end
        x = xt;
        f = ft;
        g = nlp.gobj(x); gNorm = norm(g);
        %H = opFunction(n, n, @(v, mode) nlp.hlagprod(x, y, v));
        H = nlp.hobj(x);

        iter_history(self.iter + 1, :) = ...
          [gNorm, f, flags.niters, abs(sub_opts.shift), step];

        log_line = [sprintf('%4d  ', self.iter), ...
                    sprintf(self.fmt, iter_history(self.iter + 1, :)), ...
                    sprintf('%2d', stats.istop)];
        self.logger.info(log_line);

      end

      if gNorm <= tol
        status_msg = 'Optimal';
      else
        status_msg = 'Max iterations';
      end

      self.status = utils.SolverStatus;
      self.status.iter = self.iter;
      self.status.iter_history = iter_history(1 : self.iter + 1, :);
      self.status.x = x;
      self.status.f = f;
      self.status.kkt = gNorm;
      self.status.msg = status_msg;
    end

  end  % methods

end  % classdef
