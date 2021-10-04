from __future__ import division
# import libraries
from datetime import datetime, timedelta
import pandas as pd
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from chart_studio import plotly

# import plotly.plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go

#inititate Plotly
pyoff.init_notebook_mode()

#load our data from CSV
tx_data = pd.read_csv('query_segmentation.csv')