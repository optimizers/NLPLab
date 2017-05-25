classdef ProjRecModel < model.RecModel
    %% ProjRecModel - Extension of RecModel for solvers with projections
    %   This model provides the following methods
    %
    %       * project: project point on the feasible set.
    %
    %       * projectMixed: project point on the active face of the
    %       feasible set while ensuring that the inactive constraints
    %       remain feasible.
    %
    %       * projectActiveFace: projection point on the active face of the
    %       feasible set.
    %
    %   and the property
    %       - normJac: norm of the Jacobian of the linear constraints.
    %
    %   Note that every projection will be solved according to
    %   projOpts.aOptTol.
    %
    %   ***
    %   Make sure that NlpLab is on MATLAB's path!
    %   ***
    
    
    properties (SetAccess = private, Hidden = false)
        projSolver; % Actual solver used to solve projections

        JacJact;
        normJac;
        
        krylOpts; % Options for some linear solvers
        projActFaceSolver; % Linear solver used to solve proj. act. face
        wrapProjActFace; % Internal wrapper function
    end
    
    
    methods (Access = public)
        
        function self = ProjRecModel(crit, prec, sino, geos, ...
                projSolver, projOpts, varargin)
            %% Constructor
            % Inputs:
            %   - crit: Critere object from the Poly-LION repository
            %   - prec: Precond object from the Poly-Lion repository
            %   - sino: Sinogramme object from the Poly-Lion repository
            %   - geos: Geometrie_Series object from the Poly-Lion repo
            %   - projSolver: string containing the name of the solver that
            %   will compute the projection
            %   - projOpts: struct containing the options for projSolver
            %   - mu0: (optional) initial vector for the reconstruction
            %   problem
            %   - name: (optional) name of the nlp model
            
            % Gathering optinal arguments and setting default values
            p = inputParser;
            p.addParameter('mu0', []);
            p.addParameter('name', '');
            p.addParameter('projActFaceSolver', 'pcg');
            
            % Parsing input arguments
            p.parse(varargin{:});
            % Setting values
            mu0 = p.Results.mu0;
            name = p.Results.name;
            
            % Calling the RecModel superclass (inputs will be checked here)
            self = self@model.RecModel(crit, prec, sino, geos, ...
                'mu0', mu0, 'name', name);
            
            %% Setting up projectActiveFace solver
            self.projActFaceSolver = p.Results.projActFaceSolver;
            
            switch self.projActFaceSolver
                case 'lsqr'
                    % LSQR
                    import krylov.lsqr_spot;
                    self.wrapProjActFace = @(d, ind) self.callLsqr(d, ind);
                case 'lsmr'
                    % LSMR
                    import krylov.lsmr_spot;
                    self.wrapProjActFace = @(d, ind) self.callLsmr(d, ind);
                case 'minres'
                    % MinRes
                    import krylov.minres_spot;
                    self.wrapProjActFace = @(d, ind) ...
                        self.callMinres(d, ind);
                case 'pcg'
                    % Default to PCG
                    self.wrapProjActFace = @(d, ind) self.callPcg(d, ind);
                otherwise
                    warning(['Unrecognized projActFaceSolver,', ...
                        ' default to PCG']);
                    self.wrapProjActFace = @(d, ind) self.callPcg(d, ind);
            end
            
            % Update Krylov options for projectActiveFace linear solvers
            self.krylOpts.etol = projOpts.aOptTol;
            self.krylOpts.rtol = projOpts.aOptTol;
            self.krylOpts.atol = projOpts.aOptTol;
            self.krylOpts.btol = projOpts.aOptTol;
            self.krylOpts.itnlim = self.n;
            
            %% Setting opSpots to help with the projections
            % In our case, C'*C, or C*C', can be computed efficiently, so
            % we define a special operation.
            self.JacJact = opFunction(self.objSize, self.objSize, ...
                @(x, mode) self.jacJactWrap(x));
            
            % Pointer to the norm of C
            self.normJac = self.prec.norm();
            
            %% Setting up the projection solver
            try
                % Make sure the designated solver exists
                eval(['import solvers.', projSolver, ';']);
            catch
                error('Unrecognized solver from projSolver field');
            end
            % Load the ProjModel
            import model.ProjModel;
            
            % ProjModel is a regular LeastSquaresModel with a projection
            % operation and a modified hobjprod function.
            projModel = model.ProjModel(self.JacJact, self.Jac, ...
                self.objSize);
            
            % Building the solver and storing
            self.projSolver = eval(['solvers.', projSolver, ...
                '(projModel, projOpts);']);
        end
        
        function [xProj, solved] = project(self, xbar)
            %% Project
            % This function projects xbar on C*x >= 0, i.e. it solves
            %   min     1/2 || x - \bar{x} ||^2
            %     x     C*x >= 0
            % Input:
            %   - xbar: vector to project on C*x >= 0
            % Output:
            %   - xProj: the projection of xbar on C*x >= 0
            %   - solved: boolean flag that tells if the projection was
            %   succesful. If false, the solver which is calling this
            %   function should raise an error.
            
            % Update the point that we wish to project. In the definition
            % of LeastSquaresModel, the objective function is 
            % 0.5 * || A*x - b ||^2, hence b = -xbar;
            self.projSolver.nlp.b = -xbar;
            
            % Calling the solver to solve the problem
            self.projSolver.solve();

            solved = self.projSolver.solved;
            
            % Finding the primal variable from the dual variable
            xProj = xbar + self.Jac * self.projSolver.x;
        end
        
        function [xProj, solved] = projectMixed(self, xbar, fixed)
            %% ProjectMixed
            % This function computes the projection of x on C*x >= 0 while
            % staying on the active face at x, defined by the logical array
            % fixed. In other words, we search x such that
            %
            %   min     1/2 || x - \bar{x} ||^2
            %     x     A*C*x >= 0,
            %           B*C*x = 0,
            %
            % where B is the restriction matrix defined by "fixed" and A
            % contains the other rows of the identity. This problem is
            % highly similar to the original projection sub-problem, thus
            % the same solver and model can be reused.
            % Input:
            %   - x: vector to project : {x | A*C*x >= 0, B*C*x = 0}
            % Output:
            %   - z: the projection of x on C*x >= 0
            %   - solved: boolean flag that tells if the projection was
            %   succesful. If false, the solver which is calling this
            %   function should raise an error.
            
            % Update the point that we wish to project. In the definition
            % of LeastSquaresModel, the objective function is 
            % 0.5 * || A*x - b ||^2, hence b = -xbar;
            self.projSolver.nlp.b = -xbar;
            
            % Temporarily set the components B*z unbounded, where z are the
            % Lagrange multipliers of C*x >= 0.
            bL = self.projSolver.nlp.bL(fixed);
            jLow = self.projSolver.nlp.jLow(fixed);
            self.projSolver.nlp.bL(fixed) = -inf;
            self.projSolver.nlp.jLow(fixed) = false;
            
            % Calling the solver to solve the problem
            self.projSolver.solve();
            
            % Reset the bounds to what they were
            self.projSolver.nlp.bL(fixed) = bL;
            self.projSolver.nlp.jLow(fixed) = jLow;

            solved = self.projSolver.solved;
            
            % Finding the primal variable from the dual variable
            xProj = xbar + self.Jac * self.projSolver.x;
        end
        
        function [w, solved] = projectActiveFace(self, d, ind)
            %% ProjectActiveFace - project on the active face
            % Solves the problem
            % min   1/2 || w - d||^2
            %   w   sc (C*w)_i = 0, for i \not \in the working set
            % where C is the jacobian of the linear constraints and i
            % denotes the indices of the fixed variables. This problem has
            % an analytical solution.
            %
            % From the first order KKT conditions, we can obtain the
            % following set of equations:
            %
            %   w               = d + B*C*z
            %   (B*C*C'*B') * z = -B*C*d
            %
            % where z is the lagrange mutliplier associated with the
            % equality constraint.
            %
            % Inputs:
            %   - d: vector to project on the equality constraint set
            %   - fixed: indices not in the working set
            % Output:
            %   - w: the projection of d on B*C*w = 0
            %   - solved: boolean flag that tells if the projection was
            %   succesful. If false, the solver which is calling this
            %   function should raise an error.
            
            temp = self.wrapProjActFace(d, ind);
            w = temp.w;
            solved = temp.solved;
        end
    end
    
    methods (Access = private)
        
        function temp = callPcg(self, d, fixed)
            %% CallPcg
            % Uses PCG to solve a projection on the active face
            
            subC = self.Jac(fixed, :); % B * C
            subCCt = self.JacJact(fixed, fixed);
            
            [z, flag] = pcg(subCCt, subC*(-d), ...
                self.krylOpts.atol, self.krylOpts.itnlim);
            wProj = d + (subC' * z);
            
            % Pass a struct since this is called by a wrapper function and
            % we want to return more than one argument.
            temp.w = wProj;
            temp.solved = flag == 0;
            
        end
        
        function temp = callMinres(self, d, fixed)
            %% CallMinres
            % Uses MINRES to solve a projection on the active face
            
            subC = self.Jac(fixed, :); % B * C
            subCCt = self.JacJact(fixed, fixed);
            
            [z, flags] = krylov.minres_spot(subCCt, subC*(-d), ...
                self.krylOpts);
            wProj = d + (subC' * z);
            
            % Pass a struct since this is called by a wrapper function and
            % we want to return more than one argument.
            temp.w = wProj;
            temp.solved = flags.solved;
        end
        
        function temp = callLsqr(self, d, fixed)
            %% CallLsqr
            % Uses LSQR to solve a projection on the active face
            
            subC = self.Jac(fixed, :); % B * C
            
            [z, flags] = krylov.lsqr_spot(subC', -d, self.krylOpts);
            wProj = d + (subC' * z);
            
            % Pass a struct since this is called by a wrapper function and
            % we want to return more than one argument.
            temp.w = wProj;
            temp.solved = flags.solved;
        end
        
        function temp = callLsmr(self, d, fixed)
            %% CallLsmr
            % Uses LSMR to solve a projection on the active face
            
            subC = self.Jac(fixed, :); % B * C
            
            [z, flags] = krylov.lsmr_spot(subC', -d, self.krylOpts);
            wProj = d + (subC' * z);
            
            % Pass a struct since this is called by a wrapper function and
            % we want to return more than one argument.
            temp.w = wProj;
            temp.solved = flags.solved;
        end
        
        function x = jacJactWrap(self, x)
            %% JacJactWrap
            % Simple wrapper functions to define an opSpot for C*C', or
            % equivalently C'*C (C is symmetric). In our case, C'*C can be
            % computed efficiently, so we define this special function.
            
            x = real(self.prec.AdjointDirect(x)); % C'*C*x
            
            % Calls to self.Jac (self.gcon([])) increment the nEvalC
            % counter, but not self.JacJact. We have to do it manually.
            self.AddToEvalC(1);
        end
        
    end
    
end
