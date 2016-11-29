classdef CFlashTestModel < model.qpmodel
    %
    properties (SetAccess = private, Hidden = false)
        options;
        xSol;
        tol;
        normA;
        objSize;
        
        opInv;
        opA;
    end
    
    methods (Access = public)
        function self = CFlashTestModel(name, x0, cL, cU, bL, bU, A, c, ...
                Q, options, tol)
            %% Constructor
            
            % Calling the RecModel superclass (inputs will be checked here)
            self = self@model.qpmodel(name, x0, cL, cU, bL, bU, A, c, Q);
            % Storing the options
            self.options = options;
            self.tol = tol;
            self.normA = norm(A);
            self.objSize = length(x0);
            
            aFun = @(x, mode) self.aFun(x, mode);
            invFun = @(x, mode) self.invFun(x, mode);
            
            self.opA = opFunction(self.objSize, self.objSize, aFun);
            self.opInv = opFunction(self.objSize, self.objSize, invFun);
        end
        
        function [xSol, tSol] = Solve(self)
            %% Calling the Cflash solver to solve the convex problem
            %   Inputs:
            %   - self: <this>
            %   Output:
            %   - xSol: the solution obtained with the Cflash algorithm
            
            % Checking for required parameters in Cflash
            tic;
            xSol = Cflash(self, self.options);
            self.xSol = xSol;
            tSol = toc;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % -- Cflash methods
        % Should include: project, gradProject and eqProject
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function xProj = primalProject(self, barx)
            %% Project - projection of vector x on (linear) constraint set
            % Using the dual formulation
            % min   1/2 || x - bar{x} ||^2
            %   x   sc C*x >= 0
            % Where C is the jacobian of the linear constraints (self.A)
            
            % LSQLin inputs
            H = eye(self.objSize);
            % Removing output messages
            opt =  optimoptions('lsqlin', 'Display', 'off', ...
                'TolFun', self.tol, 'TolPCG', 1e3 * self.tol, ...
                'Algorithm', 'active-set');
            % Solving the dual problem using MATLAB's lsqlin
            xProj = lsqlin(H, barx, -self.A, -self.cL, [], [], [], [], ...
                [], opt);
        end
        
        function xProj = project(self, x)
            %% Project - projection of vector x on (linear) constraint set
            % Using the dual formulation
            % min   1/2 || C'*z + x ||^2
            %   z   sc z >= 0
            % Where C is the jacobian of the linear constraints (self.A)
            
            % LSQLin inputs
            H = self.A';
            lb = zeros(self.objSize, 1);
            % Removing output messages
            opt =  optimoptions('lsqlin', 'Display', 'off', ...
                'TolFun', self.tol, 'TolPCG', 1e3 * self.tol, ...
                'Algorithm', 'active-set');
            % Solving the dual problem using MATLAB's quadprog
            z = lsqlin(H, -x, [], [], [], [], lb, [], [], ...
                opt);
            % Retrieving the primal variable  from the lagrange multiplier
            xProj = x + self.A'*z;
        end
        
        function v = primalGradProject(self, g, ind)
            %% Solve the projection problem on constraint set
            % min   1/2 || v + g ||^2
            %   v   C'*v >= 0
            opt =  optimoptions('lsqlin', 'Display', 'off', ...
                'TolFun', self.tol, 'TolPCG', 1e3 * self.tol, ...
                'Algorithm', 'active-set');
            H = eye(self.objSize);
            subA = self.A(ind, :);
            subcL = self.cL(ind);
            v = lsqlin(H, -g, -subA, -subcL, [], [], [], [], [], opt);
        end
        
        function v = gradProject(self, g, ind)
            %% Solve the projection problem on constraint set
            % min   1/2 || v + g ||^2
            %   v   (C*v)_i >= 0, \for i such that (C*x)_i = 0
            % DUAL: B*C*v <=>
            % min   1/2 || C'*B'*z - g ||^2
            %   z   z >= 0
            opt =  optimoptions('lsqlin', 'Display', 'off', ...
                'TolFun', self.tol, 'TolPCG', 1e3 * self.tol, ...
                'Algorithm', 'active-set');
            
            % LSQLin inputs
            subA = self.A(ind, :); % B * C
            H = subA';
            lb = zeros(sum(ind), 1);
            
            % Solving the dual problem using MATLAB's quadprog
            z = lsqlin(H, -g, [], [], [], [], lb, [], [], ...
                opt);
            % Retrieving the primal variable  from the lagrange multiplier
            v = -g + H * z;
        end
        
        function wProj = eqProject(self, wbar, ind)
            %% EqProject - projection of vector x under equality constraint
            % Solves the problem
            % min   1/2 || w - \bar{w}||^2
            %   w   sc (C'*w)_i = 0, if (C*x)_i = l_i
            % where C is the jacobian of the linear constraints and i
            % denotes the indices of the fixed variables. This problem has
            % an analytical solution.
            
            subA = self.A(ind, :);
            z = (subA * subA') \ (subA * (-wbar));
            
            % For the unconstrained case, the solution is trivial
            wProj = wbar + (subA' * z);
        end 
    end
    
end
