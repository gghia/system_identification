function split = splitTrainVal(N, opts)
% SPLITTRAINVAL Decide which sample indices are identification vs. validation.
% Inputs: N is the number of samples, opts chooses the split method and fraction.
% Outputs: split contains idMask, valMask and a short text summary.

    if nargin < 2 || isempty(opts), opts = struct(); end
    if ~isfield(opts,'method')    || isempty(opts.method),    opts.method = 'chunk'; end
    if ~isfield(opts,'valFrac')   || isempty(opts.valFrac),   opts.valFrac = 0.25;   end
    if ~isfield(opts,'numChunks') || isempty(opts.numChunks), opts.numChunks = 4;    end

    idMask  = false(N,1);
    valMask = false(N,1);

    switch lower(opts.method)
      case 'chunk'
        % Split the ordered data into K consecutive chunks.
        K  = opts.numChunks;
        nFrac = round(opts.valFrac * K);   % how many of K chunks go into validation
        nFrac = max(1, min(K-1, nFrac));   % keep at least one chunk in each set
        edges = round(linspace(0, N, K+1));
        for j = 1:K
            idx = (edges(j)+1):edges(j+1);
            % Put whole chunks in validation, spaced roughly evenly.
            if mod(j-1, max(1, round(K/nFrac))) == 0 && sum(valMask) < opts.valFrac*N
                valMask(idx) = true;
            else
                idMask(idx) = true;
            end
        end
        % Any leftover index goes to the identification set.
        idMask(~valMask & ~idMask) = true;

      case 'interleave'
        step  = max(2, round(1/opts.valFrac));
        valIdx = 1:step:N;
        valMask(valIdx) = true;
        idMask = ~valMask;

      case 'tail'
        nVal = round(opts.valFrac * N);
        valMask(end-nVal+1:end) = true;
        idMask = ~valMask;

      otherwise
        error('data:splitTrainVal:badMethod', ...
              'opts.method must be ''chunk'', ''interleave'', or ''tail''.');
    end

    assert(sum(idMask & valMask) == 0, 'Train/val masks overlap.');
    assert(sum(idMask | valMask) == N, 'Train/val masks do not cover all samples.');

    split.idMask  = idMask;
    split.valMask = valMask;
    split.method  = opts.method;
    split.summary = sprintf( ...
        'Split (%s): %d identification + %d validation = %d total (val=%.1f%%).', ...
        opts.method, sum(idMask), sum(valMask), N, 100*sum(valMask)/N);
end
