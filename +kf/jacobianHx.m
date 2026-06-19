function Hx = jacobianHx(t, x, u) %#ok<INUSD>
% JACOBIANHX Analytic Jacobian dh/dx for the F-16 observation model.
% Inputs: x = [u v w C], t and input are unused here.
% Outputs: Hx maps small state changes to changes in [alpha_m beta_m V_m].

    uu = x(1);
    vv = x(2);
    ww = x(3);
    C  = x(4);

    rho2 = uu^2 + ww^2;
    rho  = sqrt(rho2);
    V2   = rho2 + vv^2;
    V    = sqrt(V2);
    alpha = atan(ww / uu);

    Hx = zeros(3, 4);

    % Row 1 - alpha_m = arctan(w/u) * (1 + C)
    Hx(1,1) = -ww * (1 + C) / rho2;       % dh1/du
    Hx(1,2) = 0;                          % dh1/dv
    Hx(1,3) =  uu * (1 + C) / rho2;       % dh1/dw
    Hx(1,4) = alpha;                      % dh1/dC

    % Row 2 - beta_m = arctan( v / sqrt(u^2 + w^2) )
    Hx(2,1) = -uu * vv / (rho * V2);      % dh2/du
    Hx(2,2) =  rho / V2;                  % dh2/dv
    Hx(2,3) = -vv * ww / (rho * V2);      % dh2/dw
    Hx(2,4) = 0;                          % dh2/dC

    % Row 3 - V_m = sqrt(u^2 + v^2 + w^2)
    Hx(3,1) = uu / V;                     % dh3/du
    Hx(3,2) = vv / V;                     % dh3/dv
    Hx(3,3) = ww / V;                     % dh3/dw
    Hx(3,4) = 0;                          % dh3/dC
end
