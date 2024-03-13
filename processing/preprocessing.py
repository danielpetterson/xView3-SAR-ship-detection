import glob
import json
import os
import time
import configparser
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import math
import rasterio
from rasterio.enums import Resampling

from processing.constants import BACKGROUND, FISHING, NONFISHING, NONVESSEL