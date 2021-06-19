import numpy as np
import sys
import pandas as pd

converted_data = pd.read_excel('FARMER.xlsx', header = 1)
converted_data = converted_data.replace(',','.', regex = True)
converted_data = converted_data.astype(float)
print(converted_data)