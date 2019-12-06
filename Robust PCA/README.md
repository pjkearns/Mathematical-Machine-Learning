# Robust Principal Component Analysis via Principle Component Pursuit (PCP)

### Threshold Test:
Check tio make sure soft thresholding (st.py) and singular value thresholding (svt.py) work

### Synthetic Test:
Test PCP (pcp.py) for reconstruction error.

### Lobby Video Test:
Separate out moving person from background lobby. The stationary/background portion of the video is modeled as the low-rank component, while the person walking through is the sparse component. This is an example of using robust PCA for foreground-background separation.
