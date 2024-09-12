classdef SP_Bracket_LA
properties
    ExposureSeq         (1,:)   {mustBeInteger,mustBePositive}
    UniqueExposures     (1,:)   {mustBePositive,mustBeInteger}
    NumExposureReps     (1,:)   {mustBePositive,mustBeInteger}
    TotalSeqLength      (1,1)   {mustBeNonnegative,mustBeInteger}

    SaturationThresh    (1,:)   {mustBeNonnegative,mustBeInteger}

    ProcCenter          (1,1)   logical         = true
    B_br                (:,1)   cell
    M                   (:,1)   cell
    NumDetsOrig         (:,1)   cell
    NumDetsNow          (:,1)   cell

    MLE_LUT             (1,1)   % SP_satLA_LUT or SP_satLA_FiboLUT
    FluxMLE             (:,1)   cell

    CurrentBlockIndex   (1,1)   {mustBeNonnegative,mustBeInteger}   = 0

    SaveDir             (1,1)   string          = ""
end

methods
function obj = SP_Bracket_LA(seq, thr, opts)
    arguments
        seq             (1,:)   {mustBeInteger,mustBePositive}
        thr             (1,:)   {mustBeInteger,mustBeNonnegative}
        opts.MLE_LUT                                                = []
        opts.save_dir   (1,1)   string                              = ""
    end
    obj.ExposureSeq = seq;
    obj.TotalSeqLength = sum(obj.ExposureSeq);
    obj.UniqueExposures = unique(obj.ExposureSeq);
    obj.NumExposureReps = arrayfun(@(s) sum(obj.ExposureSeq == s), obj.UniqueExposures, UniformOutput=true);

    obj.SaturationThresh = thr;

    if isempty(opts.MLE_LUT)
        opts.MLE_LUT = SP_satLA_LUT(obj.ExposureSeq, obj.SaturationThresh);
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

    % refer to the "process" method of SP_Bracket.m for the background description.
    % Here we only describe the differences with saturation look-ahead.
    N_B = numel(B);
    N_br = floor(N_B/obj.TotalSeqLength);
    if obj.ProcCenter
        [n0_bin, ~] = end_idxs_centered(floor(N_B/2), N_br*obj.TotalSeqLength);
    else
        n0_bin = 1;
    end

    obj.B_br = cell([N_br 1]);
    obj.M = cell([N_br 1]);
    obj.NumDetsOrig = cell([N_br 1]);
    obj.NumDetsNow = cell([N_br 1]);
    obj.FluxMLE = cell([N_br 1]);

    n = n0_bin;
    for i = 1:N_br
        obj.NumDetsOrig{i} = sum(cat(3, B{n:n+obj.TotalSeqLength-1}), 3);

        obj.B_br{i} = zeros([size(B{1}, [1 2]) 2*numel(obj.UniqueExposures)], "uint8");
        % inhibition pattern (M == true ==> enabled)
        obj.M{i} = true([size(B{1}, [1 2]) numel(obj.UniqueExposures)]);
        for j = 1:numel(obj.UniqueExposures)
            for k = 1:obj.NumExposureReps(j)
                B_br_curr = false([size(B{1}, [1 2])]);
                for l = 1:obj.UniqueExposures(j)
                    % long exposure but gated by M. Since M is not changed within this loop, 
                    % it operates identical to the original exposure brackets if enabled/true.
                    B_br_curr = B_br_curr | (obj.M{i}(:,:,j) & B{n});
                    n = n + 1;
                end
                % accumulate within same exposure length as before
                obj.B_br{i}(:,:,2*j-1) = obj.B_br{i}(:,:,2*j-1) + uint8(B_br_curr);
            end
            % Record #measurements; either 0 or (number of exposures of this length)
            obj.B_br{i}(:,:,2*j) = uint8(obj.M{i}(:,:,j)) .* obj.NumExposureReps(j);
            if j < numel(obj.UniqueExposures)
                % update inhibition pattern (comparison to threshold) for all but longest exposure
                obj.M{i}(:,:,j+1) = obj.M{i}(:,:,j) & (obj.B_br{i}(:,:,2*j-1) < obj.SaturationThresh(j));
            end
        end
        % Again, not strictly necessary to perform here, but convenient for analysis.
        obj.NumDetsNow{i} = sum(obj.B_br{i}(:,:,1:2:end), 3);
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
    M = obj.M; %#ok<PROP>
    NumDetsOrig = obj.NumDetsOrig; %#ok<PROP>
    NumDetsNow = obj.NumDetsNow; %#ok<PROP>
    FluxMLE = obj.FluxMLE; %#ok<PROP>
    save(fullfile(save_dir, "full_state"), "B_br", "M", "NumDetsOrig", "NumDetsNow", "FluxMLE");

    % Images separately
    B_br_vis = cellfun(@(r) array3D2seq(double(r(:,:,1:2:end))./max(eps, double(r(:,:,2:2:end)))), ...
                        obj.B_br, UniformOutput=false);
    M_vis = cellfun(@array3D2seq, obj.M, UniformOutput=false);
    frac_inh_vis = cellfun(@(orig, new) im2uint8(1 - (max(eps,new)./max(eps,orig))), obj.NumDetsOrig, obj.NumDetsNow, ...
                        UniformOutput=false);
    MLE_vis = cellfun(@(x) SP_PPD(x).^0.4, obj.FluxMLE, UniformOutput=false);
    N_br = numel(obj.B_br);
    parfor n = 1:N_br
        name_n = sprintf("%04d", n);
        for m = 1:numel(B_br_vis{n})
            imwrite(B_br_vis{n}{m}, fullfile(save_dir, sprintf("B_br_%s_%02d.png", name_n, m)));
            imwrite(im2uint8(M_vis{n}{m}), parula(), fullfile(save_dir, sprintf("M_%s_%02d.png", name_n, m)));
        end
        imwrite(frac_inh_vis{n}, autumn(), fullfile(save_dir, sprintf("frac_%s.png", name_n)));
        imwrite(MLE_vis{n}, fullfile(save_dir, sprintf("MLE_%s.png", name_n)));
    end
end

end

end
