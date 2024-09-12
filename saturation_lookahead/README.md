# Saturation look-ahead inhibition implementation

`Fig04.m` re-creates the corresponding diagram from the paper. 
Alternate settings can be explored by changing the parameters within the script.

`SingleImageExample.m` runs exposure bracketing, both with and without saturation look-ahead, on 
an example image provided within MATLAB. No motion is considered with this static image, which 
enables a simpler illustration of the basic idea of the paper.

TODO: a `BurstImageExample.m` with burst reconstruction, to reproduce Fig. 7.

Implementations of all algorithms used are in the `src` directory, with accompanying comments.
