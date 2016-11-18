varlist = ['Age','Ancillary','D_Hour', 'DOW', 'Channel', 'Group Size', 'AP','FLA','INT','LAS','Price','Child','CheapTick5','CheapTick10',5,6,7, 8, 9, 10, 11,12,13,14,15,16,17,18,19,20,21,22,23]
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
