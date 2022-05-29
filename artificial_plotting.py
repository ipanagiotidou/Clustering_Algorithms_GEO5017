import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

X =np.array([[1, 7, 3], [5, 4, 3], [2, 1, 8], [5, 2, 5]])
clustering = [0,1,2,3]

# TODO: NORMALIZE THE DATAFRAME
df = pd.DataFrame(X)
# Normalization = each attribute value / max possible value of this attribute
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(df)
df = pd.DataFrame(x_scaled)
print("df_normalized: \n", df)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X[:,0], X[:,1], X[:,2], c = clustering, cmap = 'Greens')
plt.show()

