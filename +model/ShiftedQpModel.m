classdef ShiftedQpModel < model.NlpModel
    %% ShiftedQpModel - Unconstrained shifted quadratic model
    % This model is meant to be used in MinConf_PQN ONLY. It must receive a
    % projection function, usually coming from the NlpModel passed to
    % MinConf_PQN.
    %
    % This model is a subclass of NlpModel representing the following
    % problem
    %       { min_d     c'*d +  0.5 * d'*Q*d : (unconstrained)
    % where d = p - x, x being the 'shift' applied on p
    
    
    %% Properties
    properties (Access = private, Hidden = false)
        x; % Shift vector
        c;
        Q; % Hessian
        
        % Projection function coming from the external NlpModel that this
        % model approximates
        projFun;
    end
    
    
    %% Public methods
    methods (Access = public)
        
        function self = ShiftedQpModel(name, x0, x, c, Q, projFun)
            m = length(x0);
            m2 = length(x);
            if isempty(x0) || isempty(x)
                error('x0 & x must be specified');
            elseif m ~= m2
                error('Size mismatch between x0 & x');
            end
            
            % Unconstrained quadratic model
            cL = -inf(m, 1);
            cU = inf(m, 1);
            bL = cL;
            bU = cU;
            
            self = self@model.NlpModel(name, x0, cL, cU, bL, bU);
            
            % Store other things.
            self.x = x0;
            self.c = c;
            self.Q = opFunction(m, m, @(v, mode) Q(v));
            % Keep handle on the projection function
            self.projFun = projFun;
        end
        
        function [fObj, grad, hess] = obj(self, p)
            %% Function returning the obj. func, grad. and hess.
            % { min_d     c'*d +  0.5 * d'*Q*d : (unconstrained)
            if nargout == 1
                fObj = self.fobj_local(p);
            elseif nargout == 2
                [fObj, grad] = self.fgobj_local(p);
            else % nargout == 3
                [fObj, grad] = self.fgobj_local(p);
                % Build a Spot op such that its product is
                % hessian(z) * v
                hess = self.hobj_local(p);
            end
        end
        
        function z = project(self, p)
            %% Project
            % Pass the input argument to the projection function handle
            z = self.projFun(p);
        end
        
        function f = fobj_local(self, p)
            p = p - self.x; % d = p - x, shifting
            f = self.c' * p + 0.5 * p' * self.Q * p;
        end
        
        function g = gobj_local(self, p)
            p = p - self.x; % d = p - x, shifting
            g = self.c + self.Q * p;
        end
        
        function [f, g] = fgobj_local(self, p)
            p = p - self.x; % d = p - x, shifting
            f = p' * (self.c + 0.5 * self.Q * p);
            g = self.c + self.Q * p;
        end
        
        function H = hobj_local(self, ~)
            H = self.Q;
        end
        
        function v = hobj_prod(self, ~, ~, v)
            v = self.Q * v;
        end
        
    end
    
end
