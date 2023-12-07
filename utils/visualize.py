import cv2
import numpy as np


def play_video(videos, frame_size=256, wait_time=100):
  if not isinstance(videos, list):
    videos = [videos]

  while True:
    for frame_idx in range(len(videos[0])):
      frames = [videos[i][frame_idx] for i in range(len(videos))]
      resized_frames = [cv2.resize(frame, (frame_size, frame_size)) for frame in frames]

      # Concatenate frames side by side
      concat_frames = np.concatenate(resized_frames, axis=1)

      cv2.imshow('Frame', concat_frames)
      # Press Q on keyboard to exit
      if cv2.waitKey(wait_time) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          return


def visualize_residual(vid, rec):
  residual = np.abs(vid - rec)
  return residual
