classdef SP_Bracket
properties
    ExposureSeq     (1,:)   {mustBeInteger,mustBePositive}
    UniqueExposures (1,:)   {mustBePositive,mustBeInteger}
    NumExposureReps (1,:)   {mustBePositive,mustBeInteger}
    TotalSeqLength  (1,1)   {mustBeNonnegative,mustBeInteger}

    ProcCenter      (1,1)   logical         = true
    B_br            (:,1)   cell
    NumDetsOrig     (:,1)   cell
    NumDetsNow      (:,1)   cell

    MLE_LUT         (1,1)   % SP_bracket_LUT
    FluxMLE         (:,1)   cell

    CurrentBlockIndex   (1,1)   {mustBeNonnegative,mustBeInteger}   = 0

    SaveDir         (1,1)   string      = ""
end

methods
function obj = SP_Bracket(seq, opts)
    arguments
        seq             (1,:)   {mustBeInteger,mustBePositive}
        opts.MLE_LUT                                                = []
        opts.save_dir   (1,1)   string                              = ""
    end
    obj.ExposureSeq = seq;
    obj.TotalSeqLength = sum(obj.ExposureSeq);
    obj.UniqueExposures = unique(obj.ExposureSeq);
    obj.NumExposureReps = arrayfun(@(s) sum(obj.ExposureSeq == s), obj.UniqueExposures, UniformOutput=true);

    if isempty(opts.MLE_LUT)
        opts.MLE_LUT = SP_bracket_LUT(obj.ExposureSeq);
    end
    obj.MLE_LUT = opts.MLE_LUT;

    obj.SaveDir = opts.save_dir;
    obj = obj.setup_save();
end

function obj = process(obj, B)
    arguments
        obj 
        B       (:,1)   cell
    end

    N_B = numel(B);                         % total #binary frames
    N_br = floor(N_B/obj.TotalSeqLength);   % total #bracket blocks ==> #measurements after MLE
    if obj.ProcCenter
        % take the middle part of the sequence if not perfectly aligned
        [n0_bin, ~] = end_idxs_centered(floor(N_B/2), N_br*obj.TotalSeqLength);
    else
        n0_bin = 1;
    end

    obj.B_br = cell([N_br 1]);          % bracketed measurements
    obj.NumDetsOrig = cell([N_br 1]);   % #detections in the original binary frames (for comparisons)
    obj.NumDetsNow = cell([N_br 1]);    % #detections with bracketing implemented (here emulated)
    obj.FluxMLE = cell([N_br 1]);       % MLE of flux for each block of bracketed measurements

    n = n0_bin;
    for i = 1:N_br      % each block processed independently
        % Counting the original #detections only for later comparison/analysis.
        % This is not actually a part of the method.
        obj.NumDetsOrig{i} = sum(cat(3, B{n:n+obj.TotalSeqLength-1}), 3);

        obj.B_br{i} = zeros([size(B{1}, [1 2]) numel(obj.UniqueExposures)], "uint8");
        for j = 1:numel(obj.UniqueExposures)
            for k = 1:obj.NumExposureReps(j)
                B_br_curr = false([size(B{1}, [1 2])]);
                for l = 1:obj.UniqueExposures(j)        % emulated long-exposure
                    B_br_curr = B_br_curr | B{n};
                    n = n + 1;
                end
                % accumulate within same set of exposure lengths
                obj.B_br{i}(:,:,j) = obj.B_br{i}(:,:,j) + uint8(B_br_curr);
            end
        end
        % The two computations below would typically be performed offline.
        % The first one is not even part of the method, and only done for analysis purposes.
        obj.NumDetsNow{i} = sum(obj.B_br{i}, 3);
        obj.FluxMLE{i} = obj.MLE_LUT.mle_lookup(obj.B_br{i});
    end

    obj.CurrentBlockIndex = obj.CurrentBlockIndex + 1;
end

function obj = setup_save(obj)
    if obj.SaveDir == ""
        return;
    end

    if ~isfolder(obj.SaveDir)
        mkdir(obj.SaveDir);
    end
end

function obj = save_current_state(obj)
    if obj.SaveDir == ""
        return;
    end

    save_name = sprintf("%06d", obj.CurrentBlockIndex);
    save_dir = fullfile(obj.SaveDir, save_name);
    if ~isfolder(save_dir)
        mkdir(save_dir);
    end

    B_br = obj.B_br; %#ok<PROP>
    NumDetsOrig = obj.NumDetsOrig; %#ok<PROP>
    NumDetsNow = obj.NumDetsNow; %#ok<PROP>
    FluxMLE = obj.FluxMLE; %#ok<PROP>
    save(fullfile(save_dir, "full_state"), "B_br", "NumDetsOrig", "NumDetsNow", "FluxMLE");

    % Images separately
    B_br_vis = cellfun(@(r) array3D2seq(double(r)./reshape(obj.NumExposureReps, 1, 1, [])), ...
                        obj.B_br, UniformOutput=false);
    frac_inh_vis = cellfun(@(orig, new) im2uint8(1 - (max(eps,new)./max(eps,orig))), obj.NumDetsOrig, obj.NumDetsNow, ...
                        UniformOutput=false);
    MLE_vis = cellfun(@(x) SP_PPD(x).^0.4, obj.FluxMLE, UniformOutput=false);
    N_br = numel(obj.B_br);
    parfor n = 1:N_br
        name_n = sprintf("%04d", n);
        for m = 1:numel(B_br_vis{n})
            imwrite(B_br_vis{n}{m}, fullfile(save_dir, sprintf("B_br_%s_%02d.png", name_n, m)));
        end
        imwrite(frac_inh_vis{n}, autumn(), fullfile(save_dir, sprintf("frac_%s.png", name_n)));
        imwrite(MLE_vis{n}, fullfile(save_dir, sprintf("MLE_%s.png", name_n)));
    end
end

end

end
