import platform
import sys
import tensorflow as tf
import keras


def is_gpu_active():
    print(f"Python Platform: {platform.platform()}")
    print(f"Tensor Flow Version: {tf.__version__}")
    print(f"Keras Version: {keras.__version__}")
    print()
    print(f"Python {sys.version}")
    gpu = len(tf.config.list_physical_devices('GPU'))>0
    print("GPU is", "available" if gpu else "NOT AVAILABLE")