function [P] = SP_PPD(X)
%PPD Probability of Photon Detection (PPD) for a Single-Photon (SP) sensor
	P = 1 - exp(-X);
end
