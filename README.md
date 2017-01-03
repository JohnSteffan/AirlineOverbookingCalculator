## Personal Background

During my time as an operations research analyst at Frontier Airlines, I had access to a lot of data,
that I could not fully take advantage of.  This is what motivated me to study Data Science at Galvanize.

One of the problems in particular that I took interest in was how to use passenger level data to determine the optimal overbooking level for a given upcoming flight.  If passenger level data shows a very low chance of no shows, there should not be any overbooking.

## Overview

Anytime a flight takes off with an empty seat due to a passenger not showing up for their flight (no show), the airline looses potential revenue, because an extra seat could have been sold.  The objective of this project is to discover how to recover some of this potential revenue, without bumping too many passengers.

It is believed that data on individual passengers (Age, Advanced Purchase, Ancillary Purchase,...) that have purchased tickets can greatly influence their probability of showing up.

In the event that an airline overbooks too much, and passengers have to be bumped, passengers will first be offered compansation in the form of airline flight vouchers to take a later flight.  Thus, a passenger that volunerally takes this compansation would be happier than had they taken the original flight.

## Current Method

In the current method, previous flights are looked at and the no show rate for the bottom 10th quartile is used for the overbooking rate.

Flights without historical information would not be overbooked.

Passengers who missed their original flights but were rebooked on later flights would not be considered no shows.

## EDA

If we combine the age of a passenger, and weather or not they purchase an ancillary item, we get the below graph.  The graph below shows that passengers that purchase ancillary items are more likely to show up for there flight than those who do not.  It also appears that 10 year olds are the most likely to make a flight, while 20 year olds are the least likely.

![alt text](Figs/Age2.png)

## Features

For this study the following features were pulled for a one year period for Frontier Airlines.

| Feature Name | Description |
| ------------ |:------------------------:|
| Age | Passenger Age |
| AP | Days in advance the ticket was purchased |
| PAX | The number of passengers on a reservation |
| Ancillary | Was an ancillary item purchased at time of booking (y/n) |
| Ticket Revenue | Cost of ticket |
| Flight Type | Type of flight market, International, Vegas,... |
| Flight Hour | Hour of the day that the flight departs |
|  |  |


## Flight Level Results

If we look at two flights, one of which had zero no shows, and the other which had two, the usefulness of the model can be shown.
#### Flight 86: St Louis to Cancun

| 4/21 | 4/28 |
| ------------ |:------------------------:|
| 0 no shows predicted | 1 no show predicted |
| 0 actual no shows | 2 actual no shows |

[logo]: https://github.com/JohnSteffan/AirlineOverbookingCalculator/tree/master/Figs/EngineerAge.png "text1"


## Overall Results
For this example, January to March 2016 data was used to predict April 2016 no shows.  The acutall no shows for April 2016 were then compared to the two model predictions.

In April 2016 there were a total of 28,252 no shows, which gives plenty of space for additional revenue.

Assuming $75 in additional revenue, and a $200 cost for bumping a passenger,

| | Current System | Logistic Regression |
| ------------ |:------------------------:|:------------------------:|
| Increased Passengers | 2,005 | 6,492 |
| Bumped Passengers | 111 | 505 |
| Additional Revenue | $128,175 | $385,900 |

The above table shows that using the Logistic Regression would result in 394 additional passengers being bumped, but would lead to over $250,000 in additional revenue for the month of April.
