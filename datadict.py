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
   monthlistTrain = [1, 2, 3]#148 953
   monthlistTest = [4]
   #monthlist = [5]
   #data = data[data['Month'] ==5] # 5, 6, 7 not work, 8, 9 (10 out of 371), 10 (22 out of 441), 11 (31 out of 491), 12 33/537, 1 31/576
   #2 22/617, 3 13/636, 4, 10/553
   data = data[data['Month'].isin(monthlist)]
   data = data[data['Day'].isin([1,2,3,4])]
   nflightstest = 100
   #data = data[data['Departure Station '] =='SEA']

   data['Status_ '] = (data['Status_ '] == "NoShow").astype(int)
   data['SegmentID '] = data['SegmentID '].astype(int)
   data['Fltnum '] = data['Fltnum '].astype(int)
   data['age '] = data['age '].astype(int)
   data['AP '] = data['AP '].astype(int)
   data['DepartHour '] = data['DepartHour '].astype(int)
   data['PaxOnPNR '] = data['PaxOnPNR '].astype(int)

   data['Channel '] = data['Channel '].astype(int)
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
   # data['Status_ '].groupby(data['ticketrev '].round(-1)).mean().plot(xlim = (0, 400))
   # plt.show()


   #Create a flight key so that individual flights can be looked up
   data['flt_key'] = data.apply(get_key_from_df_flt, axis = 1)
   print "\nFlt key complette"


   print "\nStarting RF"
   flights = data['Status_ '].groupby(data['flt_key']).count()
   data.drop('Fltnum ', axis = 1, inplace = True)
   data.drop('Departure Station ', axis = 1, inplace = True)
   data.drop('DDate ', axis = 1, inplace = True)

   listofflights = flights.index.values.tolist()
   random.shuffle(listofflights)


   #Set train data
   traindata = data[data['Month'].isin(monthlistTrain)]
   testdata = data[data['Month'].isin(monthlistTest)]
   flights = data['Status_ '].groupby(data['flt_key']).count()

   #Select test flights (Don't evaluate all to save computational time)
   flights = testdata['Status_ '].groupby(testdata['flt_key']).count()
   listofflights = flights.index.values.tolist()
   random.shuffle(listofflights)
   testfltlist = listofflights[:nflightstest]


   y_train = traindata['Status_ '].values
   X_train = traindata[['age ','ancfee','DepartHour ', 'DOWDepart ', 'Channel ', 'PaxOnPNR ', 'AP ','FLA','INT','LAS','ticketrev ']].values
   #y_test =  testdata['Status_ '].values
   #X_test = testdata[['age ','ancfee','DepartHour ', 'DOWDepart ', 'Channel ', 'PaxOnPNR ', 'AP ','FLA','INT','LAS','ticketrev ']].values
   #Clear data from memory
   del traindata
   del data
   gc.collect()


   model = LogisticRegression()
   model.fit(X_train, y_train)




   from sklearn.ensemble import RandomForestClassifier
   print "\nStarting RF classifier"

   #Start random forest
   clf = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=1, random_state=0)
   classifier = clf.fit(X_train, y_train)

   # Clear Memory
   del X_train
   del y_train
   gc.collect()

   # Initialize count variables
   failcount = 0
   flightcount = 0
   bumpcount = 0
   obsuggest = 0
   for flt_i in testfltlist:
       testdata1 = testdata[testdata['flt_key'] == flt_i]
       # We only care about flights with over 160 passengers booked
       if len(testdata1) > 160:
            y_test = testdata1['Status_ '].values
            X_test = testdata1[['age ','ancfee','DepartHour ', 'DOWDepart ', 'Channel ', 'PaxOnPNR ', 'AP ','FLA','INT','LAS','ticketrev ']].values
            prba = classifier.predict_proba(X_test)
            probarray = np.array(prba[:,1])

            #Bootstrap our probabilities for each passenger to get a 95 % CI
            A = []
            for p in probarray:
                A.append(np.random.binomial(1, p, 1000))
            B = sum(A)
            NoShowPred = int(np.percentile(B, 5, interpolation = 'lower'))
            print "\nFlight =", flt_i
            print "Predicted No Show", NoShowPred
            print "Actual No Show", y_test.sum()
            print "PAX total", len(prba)
            A = []
            

            probabilities = model.predict_proba(X_test)[:, 1]
            for p in probabilities:
                A.append(np.random.binomial(1, p, 1000))
            B = sum(A)
            NoShowPred2 = int(np.percentile(B, 5, interpolation = 'lower'))

            print "Predicted No Show2", NoShowPred2

            flightcount = flightcount +1
            #Suggested Over Booking
            obsuggest += NoShowPred

            # If we overbooked too much and passengers would have needed to be bumped
            if y_test.sum() < NoShowPred:#*len(prba))*1.0-1.0:
               print "FAIL!!!!"
               failcount = failcount +1
               bumpcount += y_test.sum() - NoShowPred
   print "\n Fail Count", failcount,flightcount
   print "\n Bump / OB suggest", bumpcount, obsuggest
