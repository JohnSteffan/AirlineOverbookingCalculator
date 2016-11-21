varlist = datalist#['Ancillary','D_Hour', 'DOW', 'Channel', 'Group Size', 'AP','FLA','INT','LAS','Price','Child','CheapTick5','CheapTick10',5,6,7, 8, 9, 10, 11,12,13,14,15,16,17,18,19,20,21,22,23]
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
varlist2 = []
for i in indices:
    varlist2.append(varlist[i])
plt.figure()
plt.title('Random Forest Feature Importance')
h = plt.bar(range(X_test.shape[1]), importances[indices])
#plt.margins(0.1)
plt.subplots_adjust(bottom=0.15)
xticks_pos = [0.65*patch.get_width() + patch.get_xy()[0] for patch in h]
plt.xticks(xticks_pos, varlist2, rotation=45)

plt.show()



import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

mu, sigma = np.mean(noshowhist), np.std(noshowhist)
#x = mu + sigma*np.random.randn(10000)
x = noshowhist

# the histogram of the data
n, bins, patches = plt.hist(x, 34, normed=.2, facecolor='green', alpha=.75)

# add a 'best fit' line
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=1)

plt.xlabel('No Show Count')
plt.ylabel('Frequency')
plt.title('Histogram of Actual No Shows')
#plt.axis([0, 35, 0, 1])
plt.grid(True)

plt.show()


varlist = datalist#['Ancillary','D_Hour', 'DOW', 'Channel', 'Group Size', 'AP','FLA','INT','LAS','Price','Child','CheapTick5','CheapTick10',5,6,7, 8, 9, 10, 11,12,13,14,15,16,17,18,19,20,21,22,23]
coefs = model.coef_
indices = coefs
varlist2 = []

plt.figure()
plt.title('Logistic Regression Coef')
h = plt.bar(range(len(coefs[0])),coefs[0])
#plt.margins(0.1)
plt.subplots_adjust(bottom=0.25)
xticks_pos = [0.65*patch.get_width() + patch.get_xy()[0] for patch in h]
plt.xticks(xticks_pos, varlist, rotation=90)

plt.show()





varlist = datalist2#['Ancillary','D_Hour', 'DOW', 'Channel', 'Group Size', 'AP','FLA','INT','LAS','Price','Child','CheapTick5','CheapTick10',5,6,7, 8, 9, 10, 11,12,13,14,15,16,17,18,19,20,21,22,23]
coefs = model2.coef_
indices = coefs
varlist2 = []

plt.figure()
plt.title('Logistic Regression Coef')
h = plt.bar(range(len(coefs[0])),coefs[0])
#plt.margins(0.1)
plt.subplots_adjust(bottom=0.25)
xticks_pos = [0.65*patch.get_width() + patch.get_xy()[0] for patch in h]
plt.xticks(xticks_pos, varlist, rotation=90)

plt.show()
