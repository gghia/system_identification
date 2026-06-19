function out = iekf(Z, U, dt, params)
% IEKF Iterated Extended Kalman Filter.
% Inputs: Z are measurements, U are accelerations, dt is sample time, params tunes Q/R/P0.
% Outputs: out stores filtered states, covariance diagonals, a priori innovations,
% post-fit residuals and iterations.

if nargin < 4, params = struct(); end

% Pull / fill parameters
if ~isfield(params,'Q'), params.Q = diag([1e-6 1e-6 1e-6 0]); end
if ~isfield(params,'R'), params.R = diag([2.25e-6 2.25e-6 1]); end
if ~isfield(params,'G'), params.G = eye(4); end
% inner-loop convergence settings
if ~isfield(params,'maxIter'), params.maxIter = 10; end
if ~isfield(params,'epsTol'), params.epsTol  = 1e-10; end
% held constant) every fixed-step explicit RK scheme yields the SAME
if ~isfield(params,'integrator') || isempty(params.integrator)
    params.integrator = 'rk4';
end

if ~isfield(params,'x0') || isempty(params.x0)
    % Uses the FIRST measurement (Z(1,:))
    alpha0 = Z(1,1);
    V0     = Z(1,3);
    params.x0 = [ V0; 0; V0 * tan(alpha0); 0 ];
end

if ~isfield(params,'P0') || isempty(params.P0)
    params.P0 = diag([1e-2 1e-2 1e-2 1e0]);
end

Q          = params.Q;
R          = params.R;
G          = params.G;
maxIter    = params.maxIter;
epsTol     = params.epsTol;
x0         = params.x0;
P0         = params.P0;
integrator = lower(string(params.integrator));
switch integrator
    case "euler"
        stepFn = @step_euler;
    case "rk4"
        stepFn = @step_rk4;
end

% Pre-allocate outputs
N    = size(Z, 1);
nx   = numel(x0);
nz   = size(Z, 2);
out.xhat      = zeros(N, nx);
out.Pdiag     = zeros(N, nx);
out.innov     = zeros(N, nz);
out.resid     = zeros(N, nz);
out.iters     = zeros(N, 1);
out.alphaTrue = zeros(N, 1);

% Initialise
x_hat = x0;
P     = P0;

for k = 1:N
    % PREDICTION STEP
    u_k    = U(k,:).';
    x_pred = stepFn(x_hat, u_k, dt);

    % 1b) Covariance update. Discretising the linearisation around x_hat.
    Fx_k = kf.jacobianFx(0, x_hat, u_k);             % zero here, but kept general
    [Phi_k, Gamma_k] = kf.discretize(Fx_k, G, dt);
    P_pred = Phi_k * P * Phi_k.' + Gamma_k * Q * Gamma_k.';

    % P is a covariance matrix, so in exact arithmetic P = P'.
    % Floating point roundoff can leave tiny P(i,j) ~= P(j,i) differences;
    % averaging with P' just removes that numerical asymmetry.
    P_pred = 0.5 * (P_pred + P_pred.');

    z_k    = Z(k,:).';
    x_iter = x_pred;
    used   = maxIter;        % iteration count we actually consume
    K      = zeros(nx, nz);
    H_i    = zeros(nz, nx);

    % Missing/invalid measurement row: skip correction, keep prediction.
    if any(~isfinite(z_k))
        x_hat = x_pred;
        P     = P_pred;

        out.xhat(k, :)   = x_hat.';
        out.Pdiag(k, :)  = diag(P).';
        out.innov(k, :)  = nan(1, nz);
        out.resid(k, :)  = nan(1, nz);
        out.iters(k)     = 0;
        out.alphaTrue(k) = atan(x_hat(3) / x_hat(1));
        continue;
    end

    for i = 1:maxIter
        H_i = kf.jacobianHx(0, x_iter, u_k);
        S   = H_i * P_pred * H_i.' + R;

        K   = P_pred * H_i.' / S;

        innov_i = z_k - kf.hx(0, x_iter, u_k) - H_i * (x_pred - x_iter);
        x_new   = x_pred + K * innov_i;

        if norm(x_new - x_iter) < epsTol
            x_iter = x_new;
            used   = i;
            break;
        end
        x_iter = x_new;
    end

    % COMMIT
    x_hat = x_iter;
    I4    = eye(nx);
    % Joseph form keeps the covariance PSD in finite precision.
    P = (I4 - K * H_i) * P_pred * (I4 - K * H_i).' + K * R * K.';
    P = 0.5 * (P + P.');

    % RECORD
    out.xhat(k, :)   = x_hat.';
    out.Pdiag(k, :)  = diag(P).';
    
    % Innovation reported is the EKF (i=0) innovation z_k - h(x_pred).
    out.innov(k, :)  = (z_k - kf.hx(0, x_pred, u_k)).';
    % Residual after the inner loop has converged.
    out.resid(k, :)  = (z_k - kf.hx(0, x_hat, u_k)).';
    out.iters(k)     = used;
    out.alphaTrue(k) = atan(x_hat(3) / x_hat(1));
end

out.params = params;
end

function x_next = step_euler(x, u, dt)
% A single Euler step is EXACT for this problem because f is linear in t
% Inputs: x is current state, u is held constant over dt.
% Outputs: x_next is the state after one Euler step.
x_next = x + dt * kf.fx(0, x, u);
end

function x_next = step_rk4(x, u, dt)
% Classical RK4 step. With u held constant over [t, t+dt] and f independent
% Inputs: x is current state, u is held constant over dt.
% Outputs: x_next is the state after one RK4 step.
k1 = kf.fx(0, x,                  u);
k2 = kf.fx(0, x + 0.5 * dt * k1,  u);
k3 = kf.fx(0, x + 0.5 * dt * k2,  u);
k4 = kf.fx(0, x +       dt * k3,  u);
x_next = x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4);
end
