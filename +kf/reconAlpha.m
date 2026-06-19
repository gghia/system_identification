function out = reconAlpha(alpha_m, kfOut)
% Inputs: alpha_m is the measured angle, kfOut is the IEKF output struct.
% Outputs: out gives analytic alpha, filtered alpha, Chat and Cfinal.
% RECONALPHA Reconstruct α_true from the biased α_m measurement.

    alpha_m = alpha_m(:);
    Chat    = kfOut.xhat(:, 4);

    out.alpha_analytic = alpha_m ./ (1 + Chat);
    out.alpha_filter   = kfOut.alphaTrue;
    out.Chat           = Chat;

    % This is only a reporting number for the converged C estimate.
    % I average the end of the run instead of taking the very last sample,
    % otherwise one noisy final update would decide the value I quote.
    nTail = max(1, round(0.10 * numel(Chat)));
    out.Cfinal = mean(Chat(end-nTail+1:end));
end
