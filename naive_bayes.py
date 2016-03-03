import pandas as pd
import matplotlib.pyplot as plt
import sklearn.naive_bayes as skb
import numpy as np

df = pd.read_csv('ideal_weight.csv')

for i in df.columns.values:
	 df.rename(columns={i: (i[1:-1])}, inplace=True)
	
df['sex'] = df['sex'].replace(["'Male'", "'Female'"], ["Male", "Female"])
df['sex'] = map(lambda x: 1 if x=='Male' else 0, df['sex'])
	
# plt.figure()
# # plt.hist(df['actual'], alpha=0.5, label='actual')
# # plt.hist(df['ideal'], alpha=0.5, label='ideal')
# plt.hist(df['diff'], alpha=0.5, label='difference')
# plt.legend(loc='upper right')
# plt.show()

df = df.reindex(np.random.permutation(df.index))
X = np.array(df[['actual', 'ideal', 'diff']].values.tolist()).reshape(len(df['actual']), -3)
y = np.array(df['sex'].values.tolist()).reshape(len(df['sex']), )

nb = skb.GaussianNB()
model = nb.fit(X, y)
print model.predict(np.array([145, 160, -15]).reshape(1, -1))
print model.predict(np.array([160, 145, 15]).reshape(1, -1))

