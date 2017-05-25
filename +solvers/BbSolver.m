classdef BbSolver < solvers.NlpSolver
    %% BbSolver - Barzilai-Borwein projected gradient descent (SPG)
    % Based on the Non-monotone spectral projected gradient methods on
    % convex sets by Birgin, Martinez and Raydan
    
    properties (SetAccess = private, Hidden = false)
        % Projection calls counter
        nProj;
        memory;
        maxIterLS;
        fid;
        storedObjFunc;
        suffDec;
        maxProj;
        bbFunc;
        lsFunc;
        storedAlph;
        tau;
        projLS;

    end % private properties
    
    properties (Hidden = true, Constant)
        LOG_HEADER = { ...
            '# iter', '# obj. func.', '# proj', 't', 'alpha', ...
            'f(x)', '||Pg||'};
        LOG_FORMAT = '%10s %10s %10s %15s %15s %15s %15s\n';
        LOG_BODY = '%10d %10d %10d %15.2e %15.2e %15.4e %15.4e\n';
        ALPH_MIN = 1e-10;
        ALPH_MAX = 1e5;
        SIG_1 = 0.1;
        SIG_2 = 0.9;
    end % constant properties
    
    
    methods (Access = public)
        
        function self = BbSolver(nlp, varargin)
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
            p.addParameter('memory', 7);
            p.addParameter('fid', 1);
            p.addParameter('suffDec', 1e-4);
            p.addParameter('maxProj', 1e5);
            p.addParameter('maxIterLS', 50); % Max iters for linesearch
            p.addParameter('bbType', 'bb1');
            p.addParameter('projLS', false);
            
            p.parse(varargin{:});
            
            self = self@solvers.NlpSolver(nlp, p.Unmatched);
            
            self.memory = p.Results.memory;
            self.maxIterLS = p.Results.maxIterLS;
            self.suffDec = p.Results.suffDec;
            self.maxProj = p.Results.maxProj;
            self.projLS = p.Results.projLS;
            self.fid = p.Results.fid;
            
            % Initialize non-monotone line search objective function array
            self.storedObjFunc = -inf(self.memory, 1);
            
            import utils.PrintInfo;
            
            if self.projLS
                % Projected line search, must not project d first
                % d = -alph * g
                import linesearch.nmProjSpectArmijo;
                self.lsFunc = @(x, f, g, alph) ...
                    linesearch.nmProjSpectArmijo( ...
                    self, x, f, g, -alph * g);
            else
                % Regular line search, must project d first
                % d = P[x - alph * g] - x
                import linesearch.nmSpectralArmijo;
                self.lsFunc = @(x, f, g, alph) ...
                    linesearch.nmSpectralArmijo( ...
                    self, x, f, g, self.project(x - alph * g) - x);
            end
            
            switch p.Results.bbType
                case 'bb1'
                    self.bbFunc = @(xOld, x, gOld, g) ...
                        self.bbStep(xOld, x, gOld, g);
                case 'bb2'
                    self.bbFunc = @(xOld, x, gOld, g) ...
                        self.bbStep2(xOld, x, gOld, g);
                case 'abb'
                    self.bbFunc = @(xOld, x, gOld, g) ...
                        self.abb(xOld, x, gOld, g);
                    self.tau = 0.5;
                case 'abbmin1'
                    self.bbFunc = @(xOld, x, gOld, g) ...
                        self.abbMin1(xOld, x, gOld, g);
                    self.storedAlph = inf(self.memory, 1);
                    self.tau = 0.15;
                case 'abbss'
                    self.bbFunc = @(xOld, x, gOld, g) ...
                        self.abbMin2(xOld, x, gOld, g);
                    self.storedAlph = inf(self.memory, 1);
                    self.tau = 0.5;
                otherwise
                    % Default to first BB step length
                    self.bbFunc = @(xOld, x, gOld, g) ...
                        self.bbStep(xOld, x, gOld, g);
            end
        end % constructor
        
        function self = solve(self)
            %% Solve
            
            self.solveTime = tic;
            self.nlp.resetCounters();
            % Exit flag set to 0, will exit if not 0
            self.iStop = self.EXIT_NONE;
            % Resetting the counters
            self.nProj = 0;
            self.iter = 1;
            
            % Make sure point is feasible
            x = self.project(self.nlp.x0);
            % Evaluate initial point & derivative
            [f, g] = self.nlp.obj(x);
            pgnrm = norm(self.project(x - g) - x);
            
            fOld = inf;
            
            % Relative stopping tolerance
            self.rOptTol = self.aOptTol * norm(g);
            self.rFeasTol = self.aFeasTol * abs(f);
            
            printObj = utils.PrintInfo('Bb');
            
            % Output Log
            if self.verbose >= 2
                % Printing header
                extra = containers.Map( ...
                    {'suffDec', 'maxIterLS', 'memory'}, ...
                    {self.suffDec, self.maxIterLS, self.memory});
                printObj.header(self, extra);
                fprintf(self.LOG_FORMAT, self.LOG_HEADER{:});
            end
            
            % Initial descent direction is the steepest descent
            alph = 1;
            t = 0;
            %% Main loop
            while ~self.iStop % self.iStop == 0
                
                % Printing log
                pgnrm = norm(self.project(x - g) - x);
                
                self.nObjFunc = self.nlp.ncalls_fobj + ...
                    self.nlp.ncalls_fcon;

                if self.verbose >= 2
                    self.printf(self.LOG_BODY, self.iter, ...
                        self.nObjFunc, self.nProj, t, alph, f, pgnrm);
                end
                
                % Checking stopping conditions
                if pgnrm <= self.rOptTol + self.aOptTol
                    self.iStop = self.EXIT_OPT_TOL;
                elseif abs(f - fOld) <= self.rFeasTol + self.aFeasTol
                    self.iStop = self.EXIT_FEAS_TOL;
                elseif self.nObjFunc >= self.maxEval
                    self.iStop = self.EXIT_MAX_EVAL;
                elseif self.iter >= self.maxIter
                    self.iStop = self.EXIT_MAX_ITER;
                elseif self.nProj >= self.maxProj
                    self.iStop = self.EXIT_MAX_PROJ;
                elseif toc(self.solveTime) >= self.maxRT
                    self.iStop = self.EXIT_MAX_RT;
                end
                
                if self.iStop % self.iStop ~= 0
                    break;
                end
                
                % Check function progression
                fOld = f;
                % Storing older values to compute BB step length
                xOld = x;
                gOld = g;
                
                % Update stored objective function values
                self.storedObjFunc( ...
                    mod(self.iter - 1, self.memory) + 1) = f;
                
                % Perform a non-monotone Armijo line search
                [x, f, failed, t] = self.lsFunc(x, f, g, alph);
                
                % Evaluate gradient at new x
                g = self.nlp.gobj(x);
                
                % Exit if line search failed
                if failed
                    self.iStop = self.EXIT_MAX_ITER_LS;
                    pgnrm = norm(self.project(x - g) - x);
                    break;
                end
                
                % Compute new step length according to BB rule
                alph = self.bbFunc(xOld, x, gOld, g);
                
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
        
        function z = project(self, x)
            %% Project - projecting x on the constraint set
            [z, solved] = self.nlp.project(x);
            if ~solved
                % Propagate throughout the program to exit
                self.iStop = self.EXIT_PROJ_FAILURE;
            end
            self.nProj = self.nProj + 1;
        end
        
    end % public methods
    
    
    methods (Access = private)
        
        function alph = bbStep(self, xOld, x, gOld, g)
            %% BBStepLength - Compute Barzilai-Borwein step length
            % Safeguarded alph_BB^1 = (s' * s) / (s' * y)
            s = x - xOld;
            % Denominator of Barzilai-Borwein step length
            betaBB = s' * (g - gOld);
            if betaBB < 0
                % Fall back to maximal step length
                alph = self.ALPH_MAX;
            else
                % Compute Barzilai-Borwein step length
                % y = g - gOld
                % alph_BB = (s' * s) / (s' * y)
                % Assert alph \in [alph_min, alph_max]
                alph = min(self.ALPH_MAX, ...
                    max(self.ALPH_MIN, (s' * s) / betaBB));
            end
        end % bbsteplength
        
        function alph = bbStep2(self, xOld, x, gOld, g)
            %% BBStepLength2 - Compute Barzilai-Borwein step length
            % Safeguarded alph_BB^2 = (s' * y) / (y' * y)
            y = g - gOld;
            % Denominator of Barzilai-Borwein step length
            betaBB = y' * y;
            if betaBB < 0
                % Fall back to maximal step length
                alph = self.ALPH_MAX;
            else
                % Compute Barzilai-Borwein step length
                % s = x - xOld
                % alph_BB = (s' * y) / (y' * y)
                % Assert alph \in [alph_min, alph_max]
                alph = min(self.ALPH_MAX, ...
                    max(self.ALPH_MIN, ((x - xOld)' * y) / betaBB));
            end
        end % bbsteplength
        
        function alph = abb(self, xOld, x, gOld, g)
            %% Adaptive Barzilai-Borwein step
            
            % Evaluate both step lengths
            s = x - xOld;
            y = g - gOld;
            % No safeguards applied
            alphBb1 = (s' * s) / (s' * y);
            alphBb2 = (s' * y) / (y' * y);
            
            if alphBb2 / alphBb1 < self.tau
                alph = alphBb2;
            else
                alph = alphBb1;
            end
            
        end
        
        function alph = abbMin1(self, xOld, x, gOld, g)
            %% "Min" Adaptive Barzilai-Borwein
            % According to the procedure described in Frassoldati, Zanni,
            % Zanghirati.
            
            % Evaluate both step lengths
            s = x - xOld;
            y = g - gOld;
            % No safeguards applied
            alphBb1 = (s' * s) / (s' * y);
            alphBb2 = (s' * y) / (y' * y);
            
            % Update stored alphBb2 values
            self.storedAlph(mod(self.iter - 1, self.memory) + 1) = alphBb2;
            
            if alphBb2 / alphBb1 < self.tau
                % Take the minimum value of the stored BB2 step lengths
                alph = min(self.storedAlph);
            else
                alph = alphBb1;
            end
        end
        
        function alph = abbMin2(self, xOld, x, gOld, g)
            %% "Min" Adaptive Barzilai-Borwein
            % According to the procedure described in Frassoldati, Zanni,
            % Zanghirati.
            
            % Evaluate both step lengths
            s = x - xOld;
            y = g - gOld;
            
           bet = (s' * y);
           
           if bet <= 0
               alphBb1 = self.ALPH_MAX;
               alphBb2 = self.ALPH_MAX;
           else
              alphBb1 = max([self.ALPH_MIN, ...
                  min([(s' * s) / (s' * y), self.ALPH_MAX])]);
              alphBb2 = max([self.ALPH_MIN, ...
                  min([(s' * y) / (y' * y), self.ALPH_MAX])]);
           end
            
            % Update stored alphBb2 values
            self.storedAlph(mod(self.iter - 1, self.memory) + 1) = alphBb2;
            
            if alphBb2 / alphBb1 <= self.tau
                % Take the minimum value of the stored BB2 step lengths
                alph = min(self.storedAlph);
                self.tau = self.tau * 0.9;
            else
                alph = alphBb1;
                self.tau = self.tau * 1.1;
            end
        end
        
    end % private methods
end % class