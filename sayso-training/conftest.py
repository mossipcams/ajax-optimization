from training_paths import configure
import os
import tempfile

configure()
os.environ['GPU_LOCK'] = os.path.join(tempfile.gettempdir(), 'gpu-lock-disabled-in-tests')
