import skvideo.io
from utils.array_util import *
import parameters as params
from utils.visualization_util import *


def get_video_clips(video_path):
    video = skvideo.io.vread('/Users/shubhamsingh/projects/AnomalyDetection_UCF_Crime/input/Explosion008_x264.mp4')
    video = [cv2.resize(frame, (params.input_size, params.input_size)) for frame in video]
    clips = sliding_window(video, params.frame_count, params.frame_count)
    return clips
