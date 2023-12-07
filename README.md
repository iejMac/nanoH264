# nanoH264

video compression in as few lines of code as possible

## Objective 1

Try one very simple intra-frame prediction method and one inter-frame prediction method.

Goal: 
- Achieve 10x compression of a Moving MNIST video (data/moving-mnist-1.npy)
- Keep 90% of the pixels the same or within some error

Current Constraints:
- Simple videos
- Grayscale only

## Plan

To start off we patchify our video and then we want to try to predict each macroblock. To do this we go over our "toolbox" of prediction methods and find the method which results in the lowest residual energy. Once we find this we treat a block as predicted, save the decoded version (that will be used for future blocks that might depend on it) and move on to the next macroblock

### Prediction

Intra Prediction:
- Can depend on blocks above or to the left of the given block (raster-scan order)
- Prediction modes (depends on block size, this is for 16x16 but you can get more modes for 4x4):
  - 0 vertical. Values from block above propagate downwards
  - 1 horizontal. Values from the left propagate to the right
  - 2 Mean. Values from above and below get averaged
  - 3 Plane. A linear ‘plane’ function is fitted to the upper and left-hand samples H and V. This works well in areas of smoothly-varying luminance
- If block above and block left use the same mode its likely current block will also use that mode (either this or there is some rule, idk).


Inter Prediction:
- Use decoded frame buffer
- Try to match decoded macroblocks to current macroblock and save motion vectors

General Notes:
- For each prediction, save the residual, that needs to be compressed


After the prediction phase we should have:
- Initial data (need to start from something)
- Prediction coefficients (intra prediction modes, motion vectors)
- Residuals


### Encoding
We need to encode this stuff, not sure how yet

### Decoding
Use prediciton coefficients and residuals to reconstruct data

