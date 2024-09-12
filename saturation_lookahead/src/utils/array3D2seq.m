function [seq] = array3D2seq(arr)
%ARRAY3D2SEQ
    if isempty(arr)
        seq = {};
        return;
    end
    assert(ismember(ndims(arr), [3 4]));
    seq = squeeze(num2cell(arr, 1:(ndims(arr)-1)));
end
