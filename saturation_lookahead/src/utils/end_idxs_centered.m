function [i0, i1] = end_idxs_centered(ind_center, W)
%END_IDXS_CENTERED
% For defining ranges such that the center frame remains (approximately) 
% the same, regardless of window/block sizes or subsampling ratios
    i0 = max(1, ind_center - floor(W/2));
    i1 = i0-1 + W;
end
