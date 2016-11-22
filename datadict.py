import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import random
import scipy.stats as scs
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def get_key_from_df(df_row):
   key_v1 = df_row['age ']
   key_v2 = df_row['ancfee']
   key_v3 = df_row['DepartHour ']
   key = (key_v1, key_v2, key_v3)
   return key

def get_key_from_df_flt(df_row):
   key_v1 = df_row['Fltnum ']
   key_v2 = df_row['Departure Station ']
   key_v3 = df_row['DDate ']
   key = (key_v1, key_v2, key_v3)
   return key

def get_key_from_df_flt_baseline(df_row):
   key_v1 = df_row['Fltnum ']
   key_v2 = df_row['Departure Station ']
   key = (key_v1, key_v2)
   return key

def get_prob_based_on_key(df_row, mydict):
   key = df_row['key']
   return mydict[key]


def hr_func(ts):
    return ts.hour
def st_func(ts):
    return ts.strftime("%Y-%m-%d")
def sta_func(ts):
    return 1



if __name__ == '__main__':
   # setting up data array
   dataSF = pd.read_csv('../pdata/NoShowStillFly.csv')
   testmethod = 0
   #datalist = ['ancfee', 'DOWDepart ', 'Channel ', 'PaxOnPNR ','FLA','INT','LAS','ticketrev ','Child','CheapTick5','CheapTick10',5,6,7, 8, 9, 10, 11,12,13,14,15,16,17,18,19,20,21,22,23, 'AP3','AP7','AP14','AP21','AP28','AP35','AP45']
   datalist = ['age ','ancfee', 'PaxOnPNR ','FLA','INT','LAS','CheapTick5','CheapTick10',7, 8, 10, 11,12,13,14,15,16,17,18,21,22,23,'Child','Age20','Age30','Age40','Age50','Age60','Age70']#, 'AP3','AP7','AP14','AP21','AP28','AP35','AP45']
   datalist =  ['age ','ancfee', 'PaxOnPNR ','INT','LAS','ticketrev ',7, 8, 10, 11,12,13,14,15,16,17,18,19,22,23]#['age ','ancfee', 'PaxOnPNR ','FLA','INT','LAS','CheapTick5','CheapTick10',7, 8, 10, 11,12,13,14,15,16,17,18,21,22,23,'Child','Age20','Age30','Age40','Age50','Age60','Age70']#, 'AP3','AP7','AP14','AP21','AP28','AP35','AP45']

   datalist2 = ['INT','LAS',7, 8, 10, 11,12,13,14,15,16,17,18,19,22,23]#, 'AP3','AP7','AP14','AP21','AP28','AP35','AP45']
   #datalist2 = ['age ','ancfee', 'PaxOnPNR ','FLA','INT','LAS','ticketrev ','Child',5,6,7, 8, 9, 10, 11,12,13,14,15,16,17,18,19,20,21,22,23]#, 'AP3','AP7','AP14','AP21','AP28','AP35','AP45']

   #datalist = ['AP ','age ','ancfee', 'PaxOnPNR ','ticketrev ','Child',5,6,7, 8, 9, 10, 11,12,13,14,15,16,17,18,19,20,21,22,23]#, 'AP3','AP7','AP14','AP21','AP28','AP35','AP45']

   dataSF['OrigDepartTime'] = pd.DatetimeIndex(dataSF['OrigDepartTime'], inplace = True)

   data = pd.read_csv('../pdata/BigDataFinal.csv')#,dtype={'SegmentID ': int})#,'DOWDepart ': int, 'DOWBooking ': int, 'Channel ': int, 'Fltnum ': int})#, 'PaxOnPNR ': int, 'BAG1 ': int, 'COB1 ': int, 'SEAT ': int, 'PAX ': int, 'DepartHour ': int, 'age ': int, 'AP ': int})
   #,dtype={'PID ': int, 'SegmentID ': int})#,
   # 'DOWDepart ': int, 'DOWBooking ': int, 'Channel ': int, 'Fltnum ': int, 'PaxOnPNR ': int, 'BAG1 ': int
   # , 'COB1 ': int, 'SEAT ': int, 'PAX ': int, 'DepartHour ': int, 'age ': int, 'AP ': int})
   print "\nData input complete"


   datafull = dataSF.merge(data, how = 'inner', left_on = 'SegmentID', right_on = 'SegmentID ')
   datafull['DepartHour '] = datafull['OrigDepartTime'].apply(hr_func)
   datafull['DDate '] = datafull['OrigDepartTime'].apply(st_func)
   datafull['Departure Station '] = datafull['DepartureStation.1']
   # datafull['Arrival Station ']
   datafull['Status_ '] = datafull['OrigDepartTime'].apply(sta_func)
   # datafull['Arrival Station ']
   # datafull['Arrival Station ']
   datafull['Fltnum '] = datafull['FlightNumber']
   datanew = data.append(datafull[data.columns], ignore_index = True)
   data = datanew

   import gc

   del datafull
   del dataSF
   del datanew
   gc.collect()
   print "\nData input complete"




   data.dropna()#how = 'any')
   data['Month'] = pd.DatetimeIndex(data['DDate ']).month
   data['Day'] = pd.DatetimeIndex(data['DDate ']).day
   monthlist = [1,2,3,4]
   monthlistTrain = [1,2,3]#148 953
   monthlistTest = [4]
   #monthlist = [5]
   #data = data[data['Month'] ==5] # 5, 6, 7 not work, 8, 9 (10 out of 371), 10 (22 out of 441), 11 (31 out of 491), 12 33/537, 1 31/576
   #2 22/617, 3 13/636, 4, 10/553
   data = data[data['Month'].isin(monthlist)]
   #data = data[data['Day'].isin([1,2,3,4,5,6,7])]
   #data = data[data['Day'] == 5]

   nflightstest = 5000

   data['Status_ '] = (data['Status_ '] == "NoShow").astype(int)
   data['SegmentID '] = data['SegmentID '].astype(int)
   data['Fltnum '] = data['Fltnum '].astype(int)

   data['DepartHour '] = data['DepartHour '].astype(int)
   data['PaxOnPNR '] = data['PaxOnPNR '].astype(int)
   S = pd.get_dummies(data['DepartHour '])
   data[S.columns] = S




   #data['Status_ '].groupby(data['ticketrev '].round(0)).mean().plot(xlim = (0, 40))
   #plt.show()
   #data['Status_ '].groupby(data['AP ']).mean().plot(xlim = (0, 40))
   #plt.show()

   data['Channel '] = data['Channel '].astype(int)
   data['CheapTick5'] = np.where(data['ticketrev '] < 5, 1, 0).astype(int)
   data['CheapTick10'] = np.where(data['ticketrev '] < 10, 1, 0).astype(int)
   data['AP3'] = np.where(data['AP '] < 4, 1, 0).astype(int)

   data['AP7'] = np.where((data['AP '] >= 4) & (data['AP '] <8), 1, 0).astype(int)
   data['AP14'] = np.where((data['AP '] >= 8) & (data['AP ']<15), 1, 0).astype(int)
   data['AP21'] = np.where((data['AP '] >= 15) & (data['AP ']<22), 1, 0).astype(int)
   data['AP28'] = np.where((data['AP '] >= 22) & (data['AP ']<28), 1, 0).astype(int)
   data['AP35'] = np.where((data['AP '] >= 28) & (data['AP ']<36), 1, 0).astype(int)
   data['AP45'] = np.where((data['AP '] >= 36) & (data['AP ']<500), 1, 0).astype(int)

   data['Age20'] = np.where((data['age '] >= 18) & (data['age ']<30), 1, 0).astype(int)
   data['Age30'] = np.where((data['age '] >= 30) & (data['age ']<40), 1, 0).astype(int)

   data['Age40'] = np.where((data['age '] >= 40) & (data['age ']<50), 1, 0).astype(int)
   data['Age50'] = np.where((data['age '] >= 50) & (data['age ']<60), 1, 0).astype(int)
   data['Age60'] = np.where((data['age '] >= 60) & (data['age ']<70), 1, 0).astype(int)
   data['Age70'] = np.where((data['age '] >= 70) & (data['age ']<80), 1, 0).astype(int)
   #data['Age80'] = np.where((data['age '] >= 80) & (data['age ']<90), 1, 0).astype(int)






   data['ticketrev '] = data['ticketrev '].round(-1).astype(int)



# for month 4, uing months 11 to 3, got 170 fails out of 1264, over book was 281 to 3442




   #data['ticketrev ']=data['ticketrev '].round(0).astype(int)

   #Only look at data with significant
   data = data[data['age '] > 2.5]
   data = data[data['age '] < 80.5]


   data = data[data['DepartHour '] > 4.5]
   # Make list of flight types
   fla_list = ['MIA', 'MCO', 'PBI', 'RSW', 'TPA','UST']
   int_list = ['CUN', 'PUJ', 'SJU', 'KIN', 'SJD','PVR']
   las_list = ['LAS']

   #Create variable for if an ancillary product was purchased at time of booking
   data['ancfee'] = data[['SEAT ','COB1 ','BAG1 ']].max(axis=1).astype(int)
   #Include flight types
   data['FLA'] = np.where(data['Arrival Station '].isin(fla_list) | data['Departure Station '].isin(fla_list), 1, 0)
   data['INT'] = np.where(data['Arrival Station '].isin(int_list) | data['Departure Station '].isin(int_list), 1, 0)
   data['LAS'] = np.where(data['Arrival Station '].isin(las_list) | data['Departure Station '].isin(las_list), 1, 0)
   #data = data[data['INT'] == 0]


   #Drop unneed data
   data.drop('SEAT ', axis = 1, inplace = True)
   data.drop('COB1 ', axis = 1, inplace = True)
   data.drop('SegmentID ', axis = 1, inplace = True)
   data.drop('Arrival Station ', axis = 1, inplace = True)
   data.drop('DOWBooking ', axis = 1, inplace = True)
   data.drop('Day', axis = 1, inplace = True)
   #data.drop('Month', axis = 1, inplace = True)
   #data.drop('PID ', axis = 1, inplace = True)
   data.drop('PAX ', axis = 1, inplace = True)
   data.drop('BAG1 ', axis = 1, inplace = True)
   data['Child'] = np.where(data['age '] < 17, 1, 0).astype(int)

   #data = data.astype(int)
   # dataanc = data[data['ancfee'] == 1]
   # datanoanc = data[data['ancfee'] == 0]
   # dataanc['Status_ '].groupby(dataanc['age ']).mean().plot()
   # datanoanc['Status_ '].groupby(datanoanc['age ']).mean().plot()
   # plt.show()
   # data['Status_ '].groupby(data['age ']).mean().plot()
   # plt.show()
   # data['Status_ '].groupby(data['DepartHour ']).mean().plot()
   # plt.show()

   data['age '] = data['age '].divide(80.)
   data['AP '] = data['AP '].divide(45.)
   data['ticketrev '] = data['ticketrev '].divide(400.)
   #Create a flight key so that individual flights can be looked up
   data['flt_key'] = data.apply(get_key_from_df_flt, axis = 1)
   print "\nFlt key complette"
   data['flt_key_base'] = data.apply(get_key_from_df_flt_baseline, axis = 1)



   print "\nStarting RF"
   flights = data['Status_ '].groupby(data['flt_key']).count()
   data.drop('Fltnum ', axis = 1, inplace = True)
   data.drop('Departure Station ', axis = 1, inplace = True)
   data.drop('DDate ', axis = 1, inplace = True)

   listofflights = flights.index.values.tolist()
   random.shuffle(listofflights)
   dict = {}

   #Set train data
#    if testmethod == 0:
#
#        traindata = data[data['Month'].isin(monthlistTrain)]
#        ob['flt_key_base'] = traindata['Status_ '].groupby(traindata['flt_key']).sum()
#
#        ob = traindata.groupby(traindata['flt_key'],traindata['flt_key_base']).sum(traindata['Status_ '])
#
# ob2 = traindata[['Status_ ','flt_key','flt_key_base']].groupby(['flt_key','flt_key_base']).sum()
# ob2.groupby()
#        ob2 = ob2.groupby(ob2['flt_key_base']).quantile(q=.1)
#
# ob2['(68, DEN, 2016-02-01) (68, DEN)']
#        testdata = data[data['Month'].isin(monthlistTest)]
#        flights = data['Status_ '].groupby(data['flt_key']).count()
#        print "\n Num flights =", len(flights)
#
#        #Select test flights (Don't evaluate all to save computational time)
#        flights = testdata['Status_ '].groupby(testdata['flt_key']).count()
#        listofflights = flights.index.values.tolist()
#        random.shuffle(listofflights)
#        testfltlist = listofflights[:len(listofflights)]
#        print "\n Num flights =", len(flights)
#        raise
    #    from sys import exit
    #    exit(0)
   if testmethod == 0:

       traindata = data[data['Month'].isin(monthlistTrain)]

       testdata = data[data['Month'].isin(monthlistTest)]
       flights = data['Status_ '].groupby(data['flt_key']).count()
       print "\n Num flights =", len(flights)

       #Select test flights (Don't evaluate all to save computational time)
       flights = testdata['Status_ '].groupby(testdata['flt_key']).count()
       listofflights = flights.index.values.tolist()
       flights2 = traindata['Status_ '].groupby(traindata['flt_key_base']).count()
       trainfltlist_base = flights2.index.values.tolist()

       random.shuffle(listofflights)
       testfltlist = listofflights[:len(listofflights)]
       print "\n Num flights =", len(flights)

   book = {}
   for flt_i in trainfltlist_base:
       traindata1 = traindata[traindata['flt_key_base'] == flt_i]
       ob = traindata1['Status_ '].groupby(traindata1['flt_key']).sum()
       if len(ob) > 10:
           book[flt_i] = ob.quantile(q=.1).astype(int)
       else:
           book[flt_i] = 0
       #testfltlist = listofflights[:nflightstest]


   #testfltcount = groupby()
   #from sys import exit
   #exit(0)
   if testmethod == 1:
       traindata = data[data['Month'].isin(monthlistTrain)]
       flights = traindata['Status_ '].groupby(traindata['flt_key']).count()
       listofflights = flights.index.values.tolist()
       random.shuffle(listofflights)
       testfltlist = listofflights[:nflightstest]
       trainfltlist = listofflights[nflightstest:]
       traindata = data[data['flt_key'].isin(trainfltlist)]
       testdata = data[data['flt_key'].isin(testfltlist)]




   y_train = traindata['Status_ '].values
   X_train = traindata[datalist].values
   X_train2 = traindata[datalist2].values

   #y_test =  testdata['Status_ '].values
   #X_test = testdata[['age ','ancfee','DepartHour ', 'DOWDepart ', 'Channel ', 'PaxOnPNR ', 'AP ','FLA','INT','LAS','ticketrev ']].values
   #Clear data from memory
   #X_train = traindata[['ancfee','DepartHour ', 'CheapTick10']].values

   del traindata
   del data
   gc.collect()
   from sklearn.grid_search import GridSearchCV
   from sklearn.ensemble import RandomForestRegressor

   # random_forest_grid = {'max_depth': [3, None],
   #                    'max_features': ['sqrt', 'log2', None],
   #                    'min_samples_split': [1, 2, 4],
   #                    'min_samples_leaf': [1, 2, 4],
   #                    'bootstrap': [True, False],
   #                    'n_estimators': [10, 20, 40],
   #                    'random_state': [1]}
   #
   # rf_gridsearch = GridSearchCV(RandomForestRegressor(),
   #                           random_forest_grid,
   #                           n_jobs=-1,
   #                           verbose=True,
   #                           scoring='mean_squared_error')
   # rf_gridsearch.fit(X_train, y_train)
   #
   # print "best parameters:", rf_gridsearch.best_params_
#
# {'bootstrap': True,
#  'max_depth': None,
#  'max_features': 'sqrt',
#  'min_samples_leaf': 1,
#  'min_samples_split': 1,
#  'n_estimators': 40,
#  'random_state': 1}




   #best_rf_model = rf_gridsearch.best_estimator_



   model = LogisticRegression(n_jobs = 3)
   model = model.fit(X_train, y_train)

   model2 = LogisticRegression()
   model2.fit(X_train2, y_train)
   #from sklearn.feature_selection import VarianceThreshold




   from sklearn.ensemble import RandomForestClassifier
   print "\nStarting RF classifier"

   #Start random forest
   #clf = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=1, random_state=0)
   #classifier = clf.fit(X_train, y_train)

   # Clear Memory
   del X_train
   del y_train
   gc.collect()

   # Initialize count variables
   failcount = 0
   flightcount = 0
   bumpcount = 0
   obsuggest = 0
   failcount2 = 0
   bumpcount2 = 0
   obsuggest2 = 0
   noshowhist = []
   failcount3 = 0
   bumpcount3 = 0
   obsuggest3 = 0
   totalnoshow = 0



   for flt_i in testfltlist:
       #testdata1 = testdata[testdata['flt_key'] == flt_i]
       # We only care about flights with over 160 passengers booked
       if flights[flt_i] > 160:
            testdata1 = testdata[testdata['flt_key'] == flt_i]

            y_test = testdata1['Status_ '].values
            X_test = testdata1[datalist].values

            # prba = classifier.predict_proba(X_test)
            # probarray = np.array(prba[:,1])
            #
            # #Bootstrap our probabilities for each passenger to get a 95 % CI
            # A = []
            # for p in probarray:
            #     A.append(np.random.binomial(1, p, 1000))
            # B = sum(A)
            NoShowPred = book.get(testdata1['flt_key_base'].values[0],0)
            print "\nFlight =", flt_i
            print "Predicted No Show", NoShowPred
            print "Actual No Show", y_test.sum()
            print "PAX total", flights[flt_i]
            #noshowhist.append(y_test.sum())
            A = []


            probabilities = model.predict_proba(X_test)[:, 1]
            for p in probabilities:
                A.append(np.random.binomial(1, p, 1000))
            B = sum(A)
            NoShowPred2 = int(np.percentile(B, 5, interpolation = 'lower'))

            print "Predicted No Show2", NoShowPred2
            #print "Predicted No Confidence", model.decision_function(X_test).mean()

            A = []
            X_test2 = testdata1[datalist2].values


            probabilities = model2.predict_proba(X_test2)[:, 1]
            for p in probabilities:
                A.append(np.random.binomial(1, p, 1000))
            B = sum(A)
            #NoShowPred3 = int(np.percentile(B, 5, interpolation = 'lower'))
            NoShowPred3 = min([NoShowPred,NoShowPred2])



            print "Predicted No Show3", NoShowPred3
            #print "Predicted No Confidence", model2.decision_function(X_test2).mean()
            # if model2.decision_function(X_test2).mean() < -3.6:
            #     if  NoShowPred3 >0:
            #         NoShowPred3 -= 1



            flightcount = flightcount +1
            #Suggested Over Booking
            obsuggest += NoShowPred
            obsuggest2 += NoShowPred2
            obsuggest3 += NoShowPred3


            totalnoshow += y_test.sum()




            #If we overbooked too much and passengers would have needed to be bumped
            if y_test.sum() < NoShowPred:#*len(prba))*1.0-1.0:
                print "FAIL!!!!"
                failcount = failcount +1
                bumpcount += y_test.sum() - NoShowPred
            if y_test.sum() < NoShowPred2:#*len(prba))*1.0-1.0:
                print "FAIL!!!!"
                failcount2 = failcount2 +1
                bumpcount2 += y_test.sum() - NoShowPred2
            if y_test.sum() < NoShowPred3:#*len(prba))*1.0-1.0:
                print "FAIL!!!!"
                failcount3 = failcount3 +1
                bumpcount3 += y_test.sum() - NoShowPred3
   print "\n Fail Count", failcount,flightcount
   print "\n Bump / OB suggest", bumpcount, obsuggest
   print "\n Fail Count2", failcount2,flightcount
   print "\n Bump / OB suggest2", bumpcount2, obsuggest2
   print "\n Fail Count3", failcount3,flightcount
   print "\n Bump / OB suggest3", bumpcount3, obsuggest3
   print "\n Total No Show", totalnoshow
