addpath(genpath("src"));

clearvars;

%% Files setup
[~, script_name, ~] = fileparts(mfilename("fullpath"));
output_dir = fullfile("output", script_name);
if ~isfolder(output_dir)
    mkdir(output_dir);
end

%% Parameters setup
X = hdrread("office.hdr");      % example RGB HDR image
Xg = rgb2gray(X);
mean_Xg = mean(Xg, "all");
PPP = 0.15;          % mean incident flux per binary frame per exposure (in photons/pixel)
X_scaled = X * (PPP / mean_Xg);
Xg_scaled = Xg * (PPP / mean_Xg);

Y = 1 - exp(-X_scaled);     % for color images; each channel will be processed independently
% Y = 1 - exp(-Xg_scaled);    % grayscale version

rng(1, "simdTwister");
% the exposure bracketing sequence has a total length of 54 binary frames, 
% so we take 1 set of measurements, or 10, or 100.
% N = 54;          % number of binary frames to be simulated, and then aggregated
N = 54*10;
% N = 54*100;       % (this one will take some time to run, 
                    %  largely due to the exposure bracketing implementation -- _not_ inhibition!)

B = cell([N 1]);        
parfor n = 1:N
    B{n} = rand(size(Y)) < Y;
end

% implay(cat(dims+1, B{:}));

BRACKET_SEQ = [1 1 2 3 5 8 13 21];
SAT_THRESH = [2 1 1 1 1 1];
% An efficient LUT to convert bracketed measurements to the MLE of incident flux
% (see src/SP_satLA_FiboLUT.m for details)
LA_LUT = SP_satLA_FiboLUT(BRACKET_SEQ, SAT_THRESH);

if ndims(Y) == 2 %#ok<ISMAT>
    % for extra information to be saved 
    % (individual binary frames after bracketing, per-frame inhibition patterns, etc.)
    % save_dir_br = fullfile(output_dir, "bracket");
    % save_dir_br_LA = fullfile(output_dir, "bracket_lookahead");
    save_dir_br = "";
    save_dir_br_LA = "";
    res = generate_results(B, BRACKET_SEQ, SAT_THRESH, lookahead_MLE_LUT=LA_LUT, ...
                        save_dir_brackets_standard=save_dir_br, ...
                        save_dir_brackets_lookahead=save_dir_br_LA);
else
    res_RGB = cell([3 1]);
    ch_names = ["R", "G", "B"];
    for c = 1:3
        B_c = cell([N 1]);
        parfor n = 1:N
            B_c{n} = B{n}(:,:,c);
        end
        % save_dir_br = fullfile(output_dir, "bracket", ch_names(c));
        % save_dir_br_LA = fullfile(output_dir, "bracket_lookahead", ch_names(c));
        save_dir_br = "";
        save_dir_br_LA = "";
        res_RGB{c} = generate_results(B_c, BRACKET_SEQ, SAT_THRESH, lookahead_MLE_LUT=LA_LUT, ...
                                    save_dir_brackets_standard=save_dir_br, ...
                                    save_dir_brackets_lookahead=save_dir_br_LA);
    end
    res = combine_RGB_results(res_RGB);
end

figure;
imshow(res.Y_direct.^0.4);
pause(0.5);
title("Image recovered from complete binary frame sequence");

figure;
imshow(res.Y_brackets_standard.^0.4);
pause(0.5);
title("Image recovered from exposure brackets alone");

figure;
imshow(res.Y_brackets_lookahead.^0.4);
pause(0.5);
title("Image recovered from exposure brackets + saturation look-ahead");

inhib_frac_brkt_std = 1 - (res.D_brackets_standard ./ max(eps, res.D_direct));
inhib_frac_brkt_LA = 1 - (res.D_brackets_lookahead ./ max(eps, res.D_direct));

num_colors = 16;
figure;
subplot(2, 1, 1);
imshow(inhib_frac_brkt_std);
pause(0.5);
title({"Fraction of detections prevented/inhibited", "by standard exposure brackets"});
colormap(autumn(num_colors));
colorbar();
pause(0.5);
subplot(2, 1, 2);
imshow(inhib_frac_brkt_LA);
pause(0.5);
title("Exposure bracketing + saturation look-ahead");
colormap(autumn(num_colors));
colorbar();
pause(0.5);

function res = generate_results(B, bracket_seq, sat_thresh, opts)
arguments
    B           (:,1)   cell
    bracket_seq (1,:)   {mustBePositive,mustBeInteger}
    sat_thresh  (1,:)   {mustBeNonnegative,mustBeInteger}
    opts.save_dir_brackets_standard     (1,1)   string  = ""
    opts.save_dir_brackets_lookahead    (1,1)   string  = ""
    opts.lookahead_MLE_LUT                              = []
end
    assert(ndims(B{1}) == 2); %#ok<ISMAT>
    N_B = numel(B);
    res = struct;

    %% Single recharge period/exposure time
    %   --> directly average all binary frames      (results in max. avalanches)
    B_arr = cat(3, B{:});
    res.D_direct = sum(B_arr, 3);
    res.Y_direct = (1.0/N_B) * res.D_direct;

    %% Varying recharge periods/exposure bracketing
    %   but no inter-measurement control (inhibition policy), so all measurements are always made.
    brackets = SP_Bracket(bracket_seq, save_dir=opts.save_dir_brackets_standard);
    brackets = brackets.process(B);
    % to save individual binary frames separately (does nothing if save_dir_brackets_standard == "")
    brackets = brackets.save_current_state();
    res.D_brackets_standard = sum(cat(3, brackets.NumDetsNow{:}), 3);
    % For scenes with motion the below step is replaced by burst reconstruction 
    % (Quanta Burst Photography by Ma et al., 2020). (TODO: Make a BurstReconstructionExample.m)
    X_estimated = cat(3, brackets.FluxMLE{:});
    res.Y_brackets_standard = 1 - exp(-mean(X_estimated, 3));

    %% Brackets + saturation look-ahead to remove the most inefficient (saturated) exposure captures
    lookahead = SP_Bracket_LA(bracket_seq, sat_thresh, MLE_LUT=opts.lookahead_MLE_LUT, ...
                            save_dir=opts.save_dir_brackets_lookahead);
    lookahead = lookahead.process(B);
    lookahead = lookahead.save_current_state();
    res.D_brackets_lookahead = sum(cat(3, lookahead.NumDetsNow{:}), 3);
    X_estimated = cat(3, lookahead.FluxMLE{:});
    res.Y_brackets_lookahead = 1 - exp(-mean(X_estimated, 3));
end

function res_combined = combine_RGB_results(res_indiv)
arguments
    res_indiv   (3,1)   cell
end
    res_combined = struct;

    % add #dets. across channels
    res_combined.D_direct = res_indiv{1}.D_direct + res_indiv{2}.D_direct + res_indiv{3}.D_direct;
    res_combined.D_brackets_standard = res_indiv{1}.D_brackets_standard ...
                                        + res_indiv{2}.D_brackets_standard ...
                                        + res_indiv{3}.D_brackets_standard;
    res_combined.D_brackets_lookahead = res_indiv{1}.D_brackets_lookahead ...
                                        + res_indiv{2}.D_brackets_lookahead ...
                                        + res_indiv{3}.D_brackets_lookahead;

    % merge individual channels to make the full RGB image
    res_combined.Y_direct = cat(3, res_indiv{1}.Y_direct, ...
                                    res_indiv{2}.Y_direct, ...
                                    res_indiv{3}.Y_direct);
    res_combined.Y_brackets_standard = cat(3, res_indiv{1}.Y_brackets_standard, ...
                                                res_indiv{2}.Y_brackets_standard, ...
                                                res_indiv{3}.Y_brackets_standard);
    res_combined.Y_brackets_lookahead = cat(3, res_indiv{1}.Y_brackets_lookahead, ...
                                                res_indiv{2}.Y_brackets_lookahead, ...
                                                res_indiv{3}.Y_brackets_lookahead);
end
