function [A, expList] = buildA(X, d)
% [A, expList] = olspoly.buildA(X, d)
% Inputs: X is the data matrix, d is the total polynomial degree.
% Outputs: A is the regression matrix, expList is the exponent row list.

    [N, n] = size(X);
    if d < 0
        error('olspoly:buildA:badDegree', 'd must be >= 0.');
    end

    expList = generateExponents(n, d);     % p × n
    p_count = size(expList, 1);

    A = ones(N, p_count);
    for j = 1:p_count
        e = expList(j, :);
        for k = 1:n
            if e(k) > 0
                A(:, j) = A(:, j) .* (X(:, k) .^ e(k));
            end
        end
    end
end

% Generate all multi-indices e = (e1, …, e_n) with |e| ≤ d in graded-lex.
function E = generateExponents(n, d)
% Inputs: n is number of variables, d is the maximum total degree.
% Outputs: E lists all exponent rows used as columns of A.
    % Pre-count: C(d+n, n).
    rows = nchoosek(d + n, n);
    E    = zeros(rows, n);
    row  = 0;

    % Iterate total degree 0, 1, ..., d. For each degree D, list all
    for D = 0:d
        comps = compositions(D, n);     % each row: an n-tuple summing to D
        nRows = size(comps, 1);
        E(row+1 : row+nRows, :) = comps;
        row = row + nRows;
    end

    assert(row == rows, 'Exponent list size mismatch (logic bug).');
end

function C = compositions(D, n)
% Inputs: D is the target sum, n is the number of slots.
% Outputs: C lists all nonnegative n-tuples whose rows sum to D.
    % All n-tuples of nonnegatives summing to D, lex-ordered with leading
    if n == 1
        C = D;
        return;
    end
    rows = nchoosek(D + n - 1, n - 1);
    C    = zeros(rows, n);
    row  = 0;
    for k = D : -1 : 0                     % first component = k, decreasing
        sub = compositions(D - k, n - 1);  % decompose the rest
        m   = size(sub, 1);
        C(row+1 : row+m, 1)     = k;
        C(row+1 : row+m, 2:end) = sub;
        row = row + m;
    end
end
