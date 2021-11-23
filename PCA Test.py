from pca import PCA
from seaborn import load_dataset
import seaborn as sns
import matplotlib.pyplot as plt


df = load_dataset('iris')

X = df.drop('species', axis = 1).values
y = df['species']

pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)

print('Shape of X', X.shape, X_projected.shape)

sns.scatterplot(x = X_projected[:, 0], y = X_projected[:, 1], hue = y)

plt.show()