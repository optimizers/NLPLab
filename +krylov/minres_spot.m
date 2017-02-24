function [x, flags, stats] = minres_spot(A, b, opts)

%        [x, flags, stats] = minres_spot(A, b, opts);
%
% Spot version of minres developed by Dominique Orban.
% All optional input arguments go into the 'opts' structure with the same name
% as in the original MINRES. All original output arguments go into the 'stats'
% structure with the same name as in the original MINRES.
%
% The preconditioner is assumed to be symmetric and positive definite, i.e.,
% this method is equivalent to applying the standard MINRES to the
% centrally-preconditioned system
%          L'AL y = L'b
% where LL' = inv(M) and Ly=x.
%
% A is a linear operator.
%
% opts.M is a linear operator representing the inverse of a preconditioner P.
% More precisely, the product y = M*v should return the solution of Py = v.
% By default, opts.M is the identity.

% 31 Jan 2014: Spot version created by Dominique Orban <dominique.orban@gerad.ca>
% Spot may be obtained from https://github.com/mpf/spot
%-----------------------------------------------------------------------

% The original MINRES documentation follows.
%
%        [ x, istop, itn, rnorm, Arnorm, Anorm, Acond, ynorm ] = ...
%          minres( A, b, M, shift, show, check, itnlim, rtol )
%
% minres solves the n x n system of linear equations Ax = b
% or the n x n least squares problem           min ||Ax - b||_2^2,
% where A is a symmetric matrix (possibly indefinite or singular)
% and b is a given vector.  The dimension n is defined by length(b).
%
% INPUT:
%
% "A" may be a dense or sparse matrix (preferably sparse!)
% or a function handle such that y = A(x) returns the product
% y = A*x for any given n-vector x.
%
% If M = [], preconditioning is not used.  Otherwise,
% "M" defines a positive-definite preconditioner M = C*C'.
% "M" may be a dense or sparse matrix (preferably sparse!)
% or a function handle such that y = M(x) solves the system
% My = x given any n-vector x.
%
% If shift ~= 0, minres really solves (A - shift*I)x = b
% (or the corresponding least-squares problem if shift is an
% eigenvalue of A).
%
% When M = C*C' exists, minres implicitly solves the system
%
%            P(A - shift*I)P'xbar = Pb,
%    i.e.               Abar xbar = bbar,
%    where                      P = inv(C),
%                            Abar = P(A - shift*I)P',
%                            bbar = Pb,
%
% and returns the solution      x = P'xbar.
% The associated residual is rbar = bbar - Abar xbar
%                                 = P(b - (A - shift*I)x)
%                                 = Pr.
%
% OUTPUT:
%
% x      is the final estimate of the required solution
%        after k iterations, where k is return in itn.
% istop  is a value from [-1:9] to indicate the reason for termination.
%        The reason is summarized in msg[istop+2] below.
% itn    gives the final value of k (the iteration number).
% rnorm  estimates norm(r_k)  or norm(rbar_k) if M exists.
% Arnorm estimates norm(Ar_{k-1}) or norm(Abar rbar_{k-1}) if M exists.
%        NOTE THAT Arnorm LAGS AN ITERATION BEHIND rnorm.

% Code author: Michael Saunders, SOL and ICME, Stanford University
% Contributors:Chris Paige, School of Computer Science, McGill University
%              Sou-Cheng Choi, ICME, Stanford University
%
% 02 Sep 2003: Date of Fortran 77 version, based on
%              C. C. Paige and M. A. Saunders (1975),
%              Solution of sparse indefinite systems of linear equations,
%              SIAM J. Numer. Anal. 12(4), pp. 617-629.
%
% 02 Sep 2003: ||Ar|| now estimated as Arnorm.
% 17 Oct 2003: f77 version converted to MATLAB.
% 03 Apr 2005: A must be a matrix or a function handle.
% 10 May 2009: Parameter list shortened.
%              Documentation updated following suggestions from
%              Jeffery Kline <jeffery.kline@gmail.com>
%              (author of new Python versions of minres, symmlq, lsqr).
% 06 Jul 2009: Michael Chen <mc462@cornell.edu> reports divide by zero
%              when beta = 0 (in this case it was beta_2 = 0).
%              Realized that the istop values were out of sync.
%              They should be right now.
% 02 Sep 2011: David Fong reports error in Acond when alpha1=0.
%              gmax and gmin should be initialized before itn 1.
% 02 Sep 2011: ynorm = norm(x) is now computed directly instead of
%              being updated (incorrectly).

% Known bugs:
%  1. As Jeff Kline pointed out, Arnorm = ||A r_{k-1}|| lags behind
%     rnorm = ||r_k||.  On singular systems, this means that a good
%     least-squares solution exists before Arnorm is small enough
%     to recognize it.  The solution x_{k-1} gets updated to x_k
%     (possibly a very large solution) before Arnorm shuts things
%     down the next iteration.  It would be better to keep x_{k-1}.
%------------------------------------------------------------------

%  Retrieve input arguments.
shift = 0;
show = false;
check = false;
n = size(A,1);
itnlim = 2*n;
rtol = 1.0e-12;
etol = 1.0e-6;
M = opEye(size(A));
window = 5;
x_energy_norm2 = 0;              % Squared energy norm of x.
err_vector = zeros(window,1);    % Lower bounds on direct error in energy norm.
err_lbnds = [];                  % History of values of err_lbnds.
err_lbnd_small = false;
if nargin > 2
    if isfield(opts, 'shift')
        shift = opts.shift;
    end
    if isfield(opts, 'show')
        show = opts.show;
    end
    if isfield(opts, 'print')
        show = opts.print;
    end
    if isfield(opts, 'check')
        check = opts.check;
    end
    if isfield(opts, 'itnlim')
        itnlim = opts.itnlim;
    end
    if isfield(opts, 'rtol')
        rtol = opts.rtol;
    end
    if isfield(opts, 'etol')
        etol = opts.etol;
    end
    if isfield(opts, 'window')
        window = opts.window;
    end
    if isfield(opts, 'M')
        M = opts.M;
    end
end

%  Initialize

msg = [' beta2 = 0.  If M = I, b and x are eigenvectors '   % -1
    ' beta1 = 0.  The exact solution is  x = 0       '   %  0
    ' A solution to Ax = b was found, given rtol     '   %  1
    ' A least-squares solution was found, given rtol '   %  2
    ' Reasonable accuracy achieved, given eps        '   %  3
    ' x has converged to an eigenvector              '   %  4
    ' acond has exceeded 0.1/eps                     '   %  5
    ' The iteration limit was reached                '   %  6
    ' A  does not define a symmetric operator        '   %  7
    ' M  does not define a symmetric operator        '   %  8
    ' M  does not define a pos-def preconditioner    '   %  9
    ' The truncated error is small enough, given etol']; % 10


if show
    fprintf('\n minres.m   SOL, Stanford University   Version of 02 Sep 2011')
    fprintf('\n Solution of symmetric Ax = b or (A-shift*I)x = b')
    fprintf('\n\n n      =%8g    shift =%22.14e', n,shift)
    fprintf('\n itnlim =%8g    rtol  =%10.2e\n', itnlim,rtol)
end

istop  = 0;   itn   = 0;   Anorm = 0;    Acond = 0; Arnorm = 0;
rnorm  = 0;   ynorm = 0;   done  = false;
x      = zeros(n,1);
resvec = zeros(itnlim+1,1);

%------------------------------------------------------------------
% Set up y and v for the first Lanczos vector v1.
% y  =  beta1 P' v1,  where  P = C**(-1).
% v is really P' v1.
%------------------------------------------------------------------
r1    = b;
y     = M * b;
beta1 = b'*y;

%  Test for an indefinite preconditioner.
%  If b = 0 exactly, stop with x = 0.

if beta1< 0, istop = 9;  done = true; end
if beta1==0,             done = true; end

if beta1> 0
    beta1  = sqrt(beta1);       % Normalize y to get v1 later.
    
    % See if M is symmetric.
    
    if check
        r2     = M * y;
        s      = y' *y;
        t      = r1'*r2;
        z      = abs(s-t);
        epsa   = (s+eps)*eps^(1/3);
        if z > epsa, istop = 8;  show = true;  done = true; end
    end
    
    % See if A is symmetric.
    
    if check
        w    = A * y;
        r2   = A * w;
        s    = w'*w;
        t    = y'*r2;
        z    = abs(s-t);
        epsa = (s+eps)*eps^(1/3);
        if z > epsa, istop = 7;  done  = true;  show = true;  end
    end
end

%------------------------------------------------------------------
% Initialize other quantities.
% ------------------------------------------------------------------
oldb   = 0;       beta   = beta1;   dbar   = 0;       epsln  = 0;
qrnorm = beta1;   phibar = beta1;   rhs1   = beta1;
rhs2   = 0;       tnorm2 = 0;       gmax   = 0;       gmin   = realmax;
cs     = -1;      sn     = 0;
w      = zeros(n,1);
w2     = zeros(n,1);
r2     = r1;
resvec(1) = qrnorm;

if show
    fprintf('\n\n   Itn     x(1)     Compatible    LS       norm(A)  cond(A)')
    fprintf(' gbar/|A|\n')   %%%%%% Check gbar
end

%---------------------------------------------------------------------
% Main iteration loop.
% --------------------------------------------------------------------
if ~done                              % k = itn = 1 first time through
    while itn < itnlim
        itn    = itn+1;
        
        %-----------------------------------------------------------------
        % Obtain quantities for the next Lanczos vector vk+1, k = 1, 2,...
        % The general iteration is similar to the case k = 1 with v0 = 0:
        %
        %   p1      = Operator * v1  -  beta1 * v0,
        %   alpha1  = v1'p1,
        %   q2      = p2  -  alpha1 * v1,
        %   beta2^2 = q2'q2,
        %   v2      = (1/beta2) q2.
        %
        % Again, y = betak P vk,  where  P = C**(-1).
        % .... more description needed.
        %-----------------------------------------------------------------
        s = 1/beta;                 % Normalize previous vector (in y).
        v = s*y;                    % v = vk if P = I
        
        y = A * v - shift * v;
        if itn >= 2, y = y - (beta/oldb)*r1; end
        
        alfa   = v'*y;              % alphak
        y      = (- alfa/beta)*r2 + y;
        r1     = r2;
        r2     = y;
        y      = M * r2;
        oldb   = beta;              % oldb = betak
        beta   = r2'*y;             % beta = betak+1^2
        if beta < 0, istop = 9; break;  end
        beta   = sqrt(beta);
        tnorm2 = tnorm2 + alfa^2 + oldb^2 + beta^2;
        
        if itn==1                   % Initialize a few things.
            if beta/beta1 <= 10*eps   % beta2 = 0 or ~ 0.
                istop = -1;             % Terminate later.
            end
        end
        
        % Apply previous rotation Qk-1 to get
        %   [deltak epslnk+1] = [cs  sn][dbark    0   ]
        %   [gbar k dbar k+1]   [sn -cs][alfak betak+1].
        
        oldeps = epsln;
        delta  = cs*dbar + sn*alfa; % delta1 = 0         deltak
        gbar   = sn*dbar - cs*alfa; % gbar 1 = alfa1     gbar k
        epsln  =           sn*beta; % epsln2 = 0         epslnk+1
        dbar   =         - cs*beta; % dbar 2 = beta2     dbar k+1
        root   = norm([gbar dbar]);
        Arnorm = phibar*root;       % ||Ar{k-1}||
        
        % Compute the next plane rotation Qk
        
        gamma  = norm([gbar beta]); % gammak
        gamma  = max([gamma eps]);
        cs     = gbar/gamma;        % ck
        sn     = beta/gamma;        % sk
        phi    = cs*phibar ;        % phik
        phibar = sn*phibar ;        % phibark+1
        
        % Update  x.
        
        denom = 1/gamma;
        w1    = w2;
        w2    = w;
        w     = (v - oldeps*w1 - delta*w2)*denom;
        x     = x + phi*w;
        x_energy_norm2 = x_energy_norm2 + phi*phi;
        
        % See if lower bound on direct error has converged.
        
        err_vector(mod(itn,window)+1) = phi;
        if itn >= window
            err_lbnd = norm(err_vector);
            err_lbnds = [err_lbnds ; err_lbnd];
            err_lbnd_small = (err_lbnd <= etol * sqrt(x_energy_norm2));
        end
        
        % Go round again.
        
        gmax   = max([gmax gamma]);
        gmin   = min([gmin gamma]);
        z      = rhs1/gamma;
        rhs1   = rhs2 - delta*z;
        rhs2   =      - epsln*z;
        
        % Estimate various norms.
        
        Anorm  = sqrt( tnorm2 );
        ynorm  = norm(x);
        epsa   = Anorm*eps;
        epsx   = Anorm*ynorm*eps;
        epsr   = Anorm*ynorm*rtol;
        diag   = gbar;
        if diag==0, diag = epsa; end
        
        qrnorm = phibar;
        rnorm  = qrnorm;
        test1  = rnorm/(Anorm*ynorm);    %  ||r|| / (||A|| ||x||)
        test2  = root / Anorm;      % ||Ar{k-1}|| / (||A|| ||r_{k-1}||)
        resvec(itn+1) = qrnorm;
        
        % Estimate  cond(A).
        % In this version we look at the diagonals of  R  in the
        % factorization of the lower Hessenberg matrix,  Q * H = R,
        % where H is the tridiagonal matrix from Lanczos with one
        % extra row, beta(k+1) e_k^T.
        
        Acond  = gmax/gmin;
        
        % See if any of the stopping criteria are satisfied.
        % In rare cases, istop is already -1 from above (Abar = const*I).
        
        if istop==0
            t1 = 1 + test1;      % These tests work if rtol < eps
            t2 = 1 + test2;
            if t2    <= 1      , istop = 2; end
            if t1    <= 1      , istop = 1; end
            
            if itn   >= itnlim , istop = 6; end
            if Acond >= 0.1/eps, istop = 4; end
            if epsx  >= beta1  , istop = 3; end
            %if rnorm <= epsx   , istop = 2; end
            %if rnorm <= epsr   , istop = 1; end
            if test2 <= rtol   , istop = 2; end
            if test1 <= rtol   , istop = 1; end
            if err_lbnd_small  , istop = 10; end
        end
        
        % See if it is time to print something.
        
        prnt   = false;
        if n      <= 40       , prnt = true; end
        if itn    <= 10       , prnt = true; end
        if itn    >= itnlim-10, prnt = true; end
        if mod(itn,10)==0     , prnt = true; end
        if qrnorm <= 10*epsx  , prnt = true; end
        if qrnorm <= 10*epsr  , prnt = true; end
        if Acond  <= 1e-2/eps , prnt = true; end
        if istop  ~=  0       , prnt = true; end
        
        if show & prnt
            if mod(itn,10)==0, disp(' '); end
            str1 = sprintf('%6g %12.5e %10.3e', itn,x(1),test1);
            str2 = sprintf(' %10.3e',           test2);
            str3 = sprintf(' %8.1e %8.1e',      Anorm,Acond);
            str4 = sprintf(' %8.1e',            gbar/Anorm);
            str  = [str1 str2 str3 str4];
            fprintf('\n %s', str)
            
            debug = false;  % true;
            if debug   % Print true Arnorm.
                vv = b - A * x  + shift*x;    % vv = b - (A - shift*I)*x
                ww =     A * vv - shift*vv;   % ww = (A - shift*I)*vv = "Ar"
                trueArnorm = norm(ww);
                fprintf('\n Arnorm = %12.4e   True ||Ar|| = %12.4e', Arnorm,trueArnorm)
            end
        end % show & prnt
        
        if istop ~= 0, break; end
        
    end % main loop
end % if ~done early

% Display final status.

if show
    fprintf('\n\n istop   =%3g               itn   =%6g', istop,itn  )
    fprintf('\n Anorm   =%12.4e      Acond =%12.4e', Anorm,Acond)
    fprintf('\n rnorm   =%12.4e      ynorm =%12.4e', rnorm,ynorm)
    fprintf('\n Arnorm  =%12.4e\n', Arnorm)
    disp(msg(istop+2,:))
end

% Collect statistics.
stats.istop  = istop;
stats.Anorm  = Anorm;
stats.Acond  = Acond;
stats.rnorm  = rnorm;
stats.ynorm  = ynorm;
stats.Arnorm = Arnorm;
stats.err_lbnds = err_lbnds;
stats.x_energy_norm = sqrt(x_energy_norm2);
stats.status = msg(istop+2,:);
stats.resvec = resvec(1:itn+1);
flags.solved = (istop >= 1 & istop <= 4) | istop == 10;
flags.niters = itn;
%-----------------------------------------------------------------------
% End function minres.m
%-----------------------------------------------------------------------
end