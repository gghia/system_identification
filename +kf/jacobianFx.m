function Fx = jacobianFx(t, x, u) %#ok<INUSD>
% JACOBIANFX df/dx at (t, x, u).
% Inputs: t, x and u define the linearisation point, but f does not use x.
% Outputs: Fx is the 4 by 4 state Jacobian, zero for this model.

    Fx = zeros(4, 4);
end
