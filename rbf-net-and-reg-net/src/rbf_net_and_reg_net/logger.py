import sys
from logging import INFO, FileHandler, StreamHandler, getLogger
from pathlib import Path

def get_results_path():
    logdir = Path("./log/")
    
    if not logdir.is_dir():
        logdir.mkdir()
    
    return Path(f"{logdir}/results.txt")
    
logger = getLogger("lab5")
logger.setLevel(INFO)
logger.addHandler(StreamHandler(sys.stdout))
logger.addHandler(FileHandler(get_results_path()))
