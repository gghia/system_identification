function dx = fx(~, ~, u) 
% FX Continuous-time state-derivative
% Inputs: u = [udot vdot wdot], state is unused for this simple model.
% Outputs: dx = [udot vdot wdot 0], so C stays constant.
    dx = [ u(1);
           u(2);
           u(3);
           0     ];
end
