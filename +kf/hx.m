function z = hx(t, x, u) %#ok<INUSD>
% HX Observation equation z = h(x)
% Inputs: x = [u v w C], t and input are unused here.
% Outputs: z = [alpha_m beta_m V_m] predicted by the measurement model.

    uu = x(1);
    vv = x(2);
    ww = x(3);
    C  = x(4);

    alpha_true = atan(ww / uu);
    beta_true  = atan(vv / sqrt(uu^2 + ww^2));
    V_true     = sqrt(uu^2 + vv^2 + ww^2);

    z = [ alpha_true * (1 + C);
          beta_true;
          V_true                ];
end
