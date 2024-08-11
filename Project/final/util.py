# Add your import statements here
import nltk
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
import nltk
import numpy as np
import pandas as pd

from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import wordnet

import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
plt.style.use(['ieee','no-latex'])
plt.rcParams['figure.figsize'] = (10,5)
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['figure.titlesize'] = 20


# Add any utility functions here

