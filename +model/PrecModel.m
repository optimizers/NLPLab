classdef PrecModel < model.NlpModel
    %% PrecModel
    %  A class to test preconditioning operations
    %  
    %  A preconditioned model must contain the following methods
    %
    %  - precTimes
    %  - precDiv
    %  - precSubTimes
    %  - precSubDiv
    %
    %  The preconditioner is supposed to be an approximation of the inverse
    %  hessian of the objective function, as in the algorithm, a direction
    %  d might be replaced by a direction precTimes(d)
    
    properties
        M; % The preconditioner (H0)
        Minv; % The inverse preconditioner (B0)
    end
    
    methods
        function o = PrecModel(name, x0, cL, cU, bL, bU, M, Minv)
            %% Constructor
            o = o@model.NlpModel(name, x0, cL, cU, bL, bU);
            
            % Add preconditioner
            o.n = length(x0);
            o.M = M;
            if nargin >= 8
                o.Minv = Minv;
            else
                o.Minv = [];
            end
        end            
        
        function z = precTimes(o, z)
            z = o.M * z;
        end
        
        function z = precDiv(o, z)
            if isempty(o.Minv)
                z = o.M \ z;
            else
                z = o.Minv * z;
            end
        end
        
        function zs = precSubTimes(o, zs, ind)
            zs = o.M(ind,ind) * zs;
        end
        
        function zs = precSubDiv(o, zs, ind)
            % Warning: the result is not the same in both cases
            if isempty(o.Minv)
                zs = o.M(ind,ind) \ zs;
            else
                zs = o.Minv(ind,ind) * zs;
            end
        end
    end
end

