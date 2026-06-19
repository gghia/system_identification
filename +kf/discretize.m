function [Phi, Gamma] = discretize(Fx, G, dt)
% DISCRETIZE Convert (Fx, G) to (Phi, Gamma) for a one-step prediction.
% Inputs: Fx is the continuous Jacobian, G injects process noise, dt is sample time.
% Outputs: Phi propagates the state, Gamma propagates the held input/noise.

    n = size(Fx, 1);
    p = size(G, 2);

    if any(size(Fx) ~= [n, n])
        error('kf:discretize:badFx', 'Fx must be square; got %dx%d.', size(Fx,1), size(Fx,2));
    end
    if size(G,1) ~= n
        error('kf:discretize:badG',  'G must have n rows; got %dx%d.', size(G,1), size(G,2));
    end

    % Exact discretisation of xdot = Fx*x + G*w over one sample.
    % Here w is assumed constant during dt, same idea as c2d(...,'zoh').
    % Solving the linear system gives:
    %   x(k+1) = Phi*x(k) + Gamma*w(k)
    %   Phi    = exp(Fx*dt)
    %   Gamma  = integral_0^dt exp(Fx*tau)*G dtau
    % The augmented expm below gives both blocks at once.
    M = [Fx, G;
         zeros(p, n + p)];

    eM = expm(M * dt);

    Phi   = eM(1:n,1:n);
    Gamma = eM(1:n, n+1 : n+p);
end
