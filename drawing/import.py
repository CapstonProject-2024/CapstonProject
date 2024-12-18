import cv2
import threading
import time
import subprocess
import numpy as np
import mediapipe as mp
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from dtaidistance import dtw
