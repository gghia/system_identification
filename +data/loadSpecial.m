function S = loadSpecial(dataPath)
% LOADSPECIAL Load the 100-point special-validation grid.
% Inputs: dataPath is the assignment folder; if empty, use the current project.
% Outputs: S contains alpha_val, beta_val, Cm_val and the number of points.

    if nargin < 1 || isempty(dataPath)
        thisFile = mfilename('fullpath');
        thisDir  = fileparts(thisFile);
        codeDir  = fileparts(thisDir);
        dataPath = fileparts(codeDir);
    end

    matFile = fullfile(dataPath, 'F16validationdata_2026.mat');
    if ~isfile(matFile)
        error('data:loadSpecial:fileNotFound', ...
              'Could not find %s.', matFile);
    end

    L = load(matFile, 'Cm_val', 'alpha_val', 'beta_val');

    S.alpha_val = L.alpha_val(:);
    S.beta_val  = L.beta_val(:);
    S.Cm_val    = L.Cm_val(:);
    S.M         = numel(S.Cm_val);

    assert(numel(S.alpha_val) == S.M, 'alpha_val/Cm_val size mismatch.');
    assert(numel(S.beta_val)  == S.M, 'beta_val/Cm_val size mismatch.');
end
