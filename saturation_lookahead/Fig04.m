clearvars;

%% Files setup
[~, script_name, ~] = fileparts(mfilename("fullpath"));
output_dir = fullfile("output", script_name);
if ~isfolder(output_dir)
    mkdir(output_dir);
end

%% Parameter settings and corresponding metrics
flux = logspace(2, 6, 100);     % in photons/second
T_CR = [1e-5, 1e-4, 1e-3];      % clocked recharge intervals
N = 10;                         % #measurements
SNRH2 = cell([numel(T_CR) 1]);  % exposure-referred SNR
D = cell([numel(T_CR) 1]);      % #detections

color_CR = {"#d94701", "#fd8d3c", "#fdbe85"};
for r = numel(T_CR):-1:1
    Tr = T_CR(r);
    Hr = flux*Tr;
    Yr = 1 - exp(-Hr);  % probability of photon detection in single-photon sensor
    D{r} = N*Yr;
    SNRH2{r} = ((Hr.^2) * N) ./ (exp(Hr) - 1);  % Eq. (5) in paper
    % to re-generate plots from the paper
    if r == numel(T_CR)
        fig_SNRH2 = figure(Units="centimeters", Position=[15 15 15 10]);
        loglog(flux, SNRH2{r}, LineWidth=2, DisplayName=sprintf("CR - %.2f ms", 1e3*Tr), ...
                                Color=color_CR{r});
        ax_SNRH2 = fig_SNRH2.CurrentAxes;
        hold(ax_SNRH2, "on");
        xlabel("flux (photons/sec.)", Parent=ax_SNRH2);
        ylabel("SNR_{H}^2", Parent=ax_SNRH2);
        title(sprintf("SNR_{H}^2 with %d CR exposures", N), Parent=ax_SNRH2);

        fig_numdet = figure(Units="centimeters", Position=[15 15 15 10]);
        loglog(flux, D{r}, LineWidth=2, LineStyle="-", DisplayName=sprintf("CR - %.2f ms", 1e3*Tr), ...
                            Color=color_CR{r});
        ax_numdet = fig_numdet.CurrentAxes;
        hold(ax_numdet, "on");
        xlabel("flux (photons/sec.)", Parent=ax_numdet);
        ylabel("#detections", Parent=ax_numdet);
        title(sprintf("#det. with %d CR exposures", N), Parent=ax_numdet);
    else
        plot(ax_SNRH2, flux, SNRH2{r}, LineWidth=2, ...
                                    DisplayName=sprintf("CR - %.2f ms", 1e3*Tr), ...
                                    Color=color_CR{r});
        plot(ax_numdet, flux, D{r}, LineWidth=2, LineStyle="-", ...
                                    DisplayName=sprintf("CR - %.2f ms", 1e3*Tr), ...
                                    Color=color_CR{r});
    end
end

% Optimal _br_acket _comb_ination formula from Gnanasambandam and Chan (2020)
SNRH2_arr = cat(1, SNRH2{:});
SNRH2_sum = sum(SNRH2_arr, 1);
SNRH2_frac = SNRH2_arr ./ SNRH2_sum;
SNRH2_br_comb = 1.0 ./ sum((SNRH2_frac.^2)./SNRH2_arr, 1);
color_br_comb = "#ce1256";
plot(ax_SNRH2, flux, SNRH2_br_comb, LineWidth=5, DisplayName="comb. (normal)", Color=color_br_comb);

legend(ax_SNRH2, "Location", "southeast");
legend(ax_numdet, "Location", "northwest");
ylim(ax_SNRH2, [1e-1 1e1]);
ylim(ax_numdet, [1e0 2*N]);

% saturation look-ahead impact
sat_thr = [7 7];
prob_sat = cell([numel(T_CR) 1]);
for r = 1:numel(T_CR)-1
    Tr = T_CR(r);
    Hr = flux*Tr;    
    Yr = 1 - exp(-Hr);
    prob_sat{r} = binocdf(sat_thr(r)-1, N, Yr, "upper");
end

D_lookahead = cell([numel(T_CR) 1]);
% probability of making a measurement in that bracket, i.e., all the previous brackets not saturating
prob_meas_lookahead = cell([numel(T_CR) 1]);
SNRH2_lookahead = cell([numel(T_CR) 1]);
D_lookahead{1} = D{1};
prob_meas_lookahead{1} = ones(size(D_lookahead{1}));
SNRH2_lookahead{1} = SNRH2{1};
for r = 2:numel(T_CR)
    % The SNR_H^2 can be shown to have the same form as the one with the original brackets, 
    % except for replacing the original #dets. with 
    %                       prob_meas_lookahead{r} * #dets.             (the new effective signal).
    prob_meas_lookahead{r} = prob_meas_lookahead{r-1} .* (1 - prob_sat{r-1});
    D_lookahead{r} = D{r} .* prob_meas_lookahead{r};
    SNRH2_lookahead{r} = prob_meas_lookahead{r} .* SNRH2{r};
end
SNRH2_lookahead_arr = cat(1, SNRH2_lookahead{:});
SNRH2_lookahead_sum = sum(SNRH2_lookahead_arr, 1);
SNRH2_lookahead_frac = SNRH2_lookahead_arr ./ SNRH2_lookahead_sum;
SNRH2_lookahead_comb = 1.0 ./ sum((SNRH2_lookahead_frac.^2)./max(eps, SNRH2_lookahead_arr), 1);

color_lookahead_comb = "#d7b5d8";
plot(ax_SNRH2, flux, SNRH2_lookahead_comb, LineWidth=2, DisplayName="comb. (inhibited)", ...
                                            Color=color_lookahead_comb);
legend(ax_SNRH2, "Location", "southeast");
ylim(ax_SNRH2, [1e-1 1e1]);
for r = 1:numel(T_CR)
    Tr = T_CR(r);
    plot(ax_numdet, flux, D_lookahead{r}, LineWidth=2, LineStyle="--", ...
                                        DisplayName=sprintf("CR - %.2f ms", 1e3*Tr), ...
                                        Color=color_CR{r});
end
fpath_SNRH2_svg = fullfile(output_dir, "SNRH2.svg");
set(fig_SNRH2, Renderer="painters"); %#ok<*FGREN>
saveas(fig_SNRH2, fpath_SNRH2_svg);

fpath_numdet_svg = fullfile(output_dir, "numdet.svg");
set(fig_numdet, Renderer="painters");
saveas(fig_numdet, fpath_numdet_svg);

%% Comparing total #dets between conventional brackets and saturation lookahead (Fig. 4c)
D_arr = cat(1, D{:});
D_total = sum(D_arr, 1);
D_lookahead_arr = cat(1, D_lookahead{:});
D_lookahead_total = sum(D_lookahead_arr, 1);

fig_numdet_total = figure(Units="centimeters", Position=[15 15 15 9]);
loglog(flux, D_total, LineWidth=2, DisplayName="conv. bracket", Color=color_br_comb);
ax_numdet_total = fig_numdet_total.CurrentAxes;
hold(ax_numdet_total, "on");
plot(ax_numdet_total, flux, D_lookahead_total, LineWidth=2, DisplayName="w/ inhibition", ...
                                            Color=color_lookahead_comb);
xlabel("flux (photons/sec.)", Parent=ax_numdet_total);
ylabel("#det.", Parent=ax_numdet_total);
title({"Total #det.", "before/after inhibition"}, Parent=ax_numdet_total);

legend(ax_numdet_total, "Location", "southeast");
ylim(ax_numdet_total, [1e0 3.2*N]);

fpath_numdet_total_svg = fullfile(output_dir, "numdet_total.svg");
set(fig_numdet_total, Renderer="painters");
saveas(fig_numdet_total, fpath_numdet_total_svg);
