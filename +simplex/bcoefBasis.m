function B = bcoefBasis(b, kappa, multinomCoef)
% Inputs: b are barycentric rows, kappa/multinomCoef define the basis powers.
% Outputs: B is the basis matrix evaluated at each row of b.
% BCOEFBASIS Evaluate the de-Boor B-form basis B^d_κ(b).

    M  = size(b, 1);
    dHat = size(kappa, 1);
    nP1  = size(kappa, 2);            % n+1

    if size(b, 2) ~= nP1
        error('simplex:bcoefBasis:dimMismatch', ...
              'b has %d columns; kappa has %d columns; they must match.', ...
              size(b,2), nP1);
    end

    B = zeros(M, dHat);
    for j = 1:dHat
        ej   = kappa(j, :);             % 1×(n+1)
        % Per-row product of b(i, k)^ej(k); 0^0 trapping by skipping zero
        bij = ones(M, 1);
        for k = 1:nP1
            if ej(k) > 0
                bij = bij .* (b(:, k) .^ ej(k));
            end
        end
        B(:, j) = multinomCoef(j) * bij;
    end
end
