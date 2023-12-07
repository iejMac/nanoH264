import cv2
import numpy as np


def play_video(vid, frame_size=256, wait_time=100):
    for frame in vid:
        if frame_size is not None:
            frame = cv2.resize(frame, (frame_size, frame_size))

        cv2.imshow('Frame', frame)

        # Press Q on keyboard to exit 
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    # Closes all the frames 
    cv2.destroyAllWindows()
