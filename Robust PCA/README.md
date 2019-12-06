# Robust Principal Component Analysis via Principle Component Pursuit (PCP)

### Threshold Test:
Check tio make sure soft thresholding and singular value threshollding work

### Synthetic Test:
Test PCP for reconstruction error

### Lobby Video Test:
Separate out moving person from background lobby. The stationary/background portion of the video is modeled as the low-rank component, while the person walking through is the sparse component. You should see that the second and third plots show the background only and person only. This is an example of using robust PCA for foreground-background
separation.
