# -*- coding: utf-8 -*-
"""KF_ Python_project_Template_sara_long_track.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Vgz2HLcAVdzox_U_Nvg1KrvzvOOYc74e

# **Title of the Project**
Hotel Booking Cancellation Prediction

# **About Dataset:**
Hotel bookings dataset show the booking status for each booking for Resort and City Hotel from 2015 to 2017.

These are the columns from the dataset that are **not required **in the machine learning:
1. lead_time 
2. arrival_date_month
3. arrival_date_year
4. arrival_date_day
5. arrival_date_week
6. stay_in_weekend_nights
7. stay_in_week_nights
8. is_repeated_guest
9. previous_cancellations
10. previous_booking_not_canceled
11. agent
12. company 
13. adr
14. total_of_special_requests
15. reservation_status
16. reservation_status_date

is_canceled is the **target** column in the dataset.

Here are the **independent** columns in the dataset:
1. hotel
2. adult
3. babies
4. children
5. meal
6. country
7. market_segment
8. distribution_channel
9. reserved_room_type
10. assigned_room_type
11. booking_changes
12. deposit_type
13. days_in_waitin_list
14. customer_type
15. required_car_parking_spaces

My target column is a categorical data hence I will be using Supervised Classification Machine Learning.

# **Step 1:Import necessary libraries**
"""

#test both methods on data spliting - randow or k-4 method.  for random, more iterations
#training dataset 70: validation dataset 10: testing 20

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

"""# **Step 2:Import the data inside google colab**"""

# loading the data and setting the unique client_id as the index::

df = pd.read_csv('https://github.com/KoonFoong/Hotel_Booking_Cancellation_Prediction/blob/main/hotel_bookings.csv')
d_copy = df

"""# **Step 3:Data Exploration**"""

df.head()

df.tail()

df.shape

"""There are 119,390 rows and 32 columns in the dataset."""

df.info()

df.dtypes

df.describe()

df2=df.drop(['lead_time','arrival_date_year','arrival_date_month','arrival_date_week_number','arrival_date_day_of_month',
            'stays_in_weekend_nights','stays_in_week_nights','is_repeated_guest','previous_cancellations','previous_bookings_not_canceled','agent','company','adr','total_of_special_requests','reservation_status','reservation_status_date'],axis=1)

df2.info()

df2.head()

df2.describe()

df2.info()

"""# **Step 4:Data Cleaning**

Check for missing values,duplicate values,categorical values and outliers and handle them accordingly.

Consider the table is read using variable "df" Then, use the functions like


1.   Check for the column datas types and customise the datatype if necessary
2.   df.isna().sum()
3.   df['colname'].fillna()
4.   df[df.duplicated()]
5.   df.drop_duplicates()
6.   Use box plot to check for outliers
7.   Remove the outliers by any technique
8.   Scaling of numerical features
9.   Encode the categorical data into numerical
10.  Display the clean data


"""

#check for null values
df2.isna().sum()

"""Children and country variables have null values"""

df2=df2.dropna()

df2['children']=df2['children'].astype('int64')

df2['required_car_parking_spaces']=df2['required_car_parking_spaces'].astype('int64')

df2[df2.duplicated()]

df2.drop_duplicates()

#Finding outliers
df2.boxplot()

df2['adults'].plot(kind="box")

df2['babies'].plot(kind="box")

"""The outliers can be ignored."""

#scaling numerical values
df2.info()

# for Adults :
avg_adults = df2['adults'].mean()
avg_adults

std_adults = df2['adults'].std()
std_adults

df2['Z_Score_Adults'] = (df2['adults'] - avg_adults)/std_adults

# for Childrens :
avg_children = df2['children'].mean()
avg_children

std_children = df2['children'].std()
std_children

df2['Z_Score_Children'] = (df2['children'] - avg_children)/std_children

# for Babies :
avg_babies = df2['babies'].mean()
avg_babies

std_babies = df2['babies'].std()
std_babies

df2['Z_Score_Babies'] = (df2['babies'] - avg_babies)/std_babies

# required_car_parking_spaces :
avg_required_car_parking_spaces = df2['required_car_parking_spaces'].mean()
avg_required_car_parking_spaces

std_required_car_parking_spaces = df2['required_car_parking_spaces'].std()
std_required_car_parking_spaces

df2['Z_Score_Required_car_parking_spaces'] = (df2['required_car_parking_spaces'] - avg_required_car_parking_spaces)/std_required_car_parking_spaces

# Distribution of the columns

fig, axes = plt.subplots(2,2, figsize=(15,5))

sns.distplot(df2['adults'], ax=axes[0,0])
sns.distplot(df2['Z_Score_Adults'], ax=axes[0,1])
sns.distplot(df2['children'], ax=axes[1,0])
sns.distplot(df2['Z_Score_Children'], ax=axes[1,1])


plt.show()

# Distribution of the columns

fig, axes = plt.subplots(2,2, figsize=(15,5))

sns.distplot(df2['babies'], ax=axes[0,0])
sns.distplot(df2['Z_Score_Babies'], ax=axes[0,1])
sns.distplot(df2['required_car_parking_spaces'], ax=axes[1,0])
sns.distplot(df2['Z_Score_Required_car_parking_spaces'], ax=axes[1,1])


plt.show()

df2

#Factorize the data
for i in df2.columns:
    if df2[i].dtypes=='object':
        df2[i] = pd.Categorical(pd.factorize(df2[i])[0])

df2

df2.info()

"""# **Step 5:Data Visualization**
Explain the findings on visualizing the data
(Create atleast 5 charts using matplotlib and seaborne)and one chart using sweet Viz tool

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""**How many bookings for each hotel?**"""

a=d_copy['hotel'].value_counts()
a

ax = sns.countplot(x = d_copy['hotel']);

# set y-axis label
ax.set_ylabel("Total Bookings")

"""A bar graph is used to compare different quantities or values across different categories or groups.

Total of 79330 bookings from City Hotel and 40060 bookings from Resort Hotel.

**Booking status for each hotel**
"""

bx =sns.histplot(d_copy, x = 'is_canceled', hue = 'hotel', multiple = 'stack', palette = ['coral', 'skyblue']);

# set y-axis label
bx.set_ylabel("Total Bookings")

"""Stacked bar is used to show more than one values for each category.

In general, City Hotel had the most bookings.  More than half of the cancellation came from City Hotel.

**What is the % of successful bookings?**
"""

d_copy.groupby('is_canceled').size().plot(kind = 'pie', 
                                   autopct = '%.2f%%', 
                                   labels = ['Cancelled', 'Successful'], 
                                   label = 'Booking Status', 
                                   fontsize = 15,
                                   colors = ['red', 'green']);

"""A pie chart is used to show the distribution or composition of categorical data. 

Out of all the bookings made, only 37% were successful while the remaining 63% were cancelled.

**Which Market Segment contributed most of the Successful Bookings?**
"""

plt.bar(d_copy['market_segment'], d_copy['previous_bookings_not_canceled'], width=0.7)
plt.title('Successful Bookings based on Market Segment')
plt.xlabel('Market Segment')
plt.ylabel('Previous Bookings Not Cancelled')
plt.show()

"""A bar graph is used to compare different quantities or values across different categories or groups.

Corporate generated the most successful bookings vs Aviation with the least bookings.

**What type of room had the highest number of reservations?**
"""

c=d_copy['reserved_room_type'].value_counts()
c

cx = sns.countplot(x = d_copy['reserved_room_type']);

# set y-axis label
cx.set_ylabel("Total Reservations")

"""A bar graph is used to compare different quantities or values across different categories or groups.

From the above, Room Type A has the highest reserved rate followed by Room D and Room E.  Room L and P have the least reservations.  
"""

!pip install sweetviz
# importing sweetviz
import sweetviz as sv

#analyzing the dataset
hotel_report = sv.analyze(d_copy)

hotel_report.show_html('Hotel Booking Cancellation Prediction.html')
hotel_report.show_notebook()

"""# **Step 6: Data Splitting into train and test**"""

from sklearn.model_selection import train_test_split

df3 = df2.drop(['adults','children','babies','required_car_parking_spaces'],axis=1)

df3.info()

df3.fillna(0, inplace=True)

X=pd.DataFrame(df3.iloc[:,0:25])# df.iloc[rows,cols]
y=pd.DataFrame(df3['is_canceled'])

X.head()

y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42,train_size=0.8)

"""# **Step 7: Model Building**


"""

from sklearn.linear_model import LogisticRegression
lm1= LogisticRegression()
lm1.fit(X_train, y_train)

lm1.coef_

lm1.intercept_

ypred=lm1.predict(X_test)
ypred

y_test

"""# **Step 8: Model Validation**"""

from sklearn.metrics import accuracy_score
accuracy_score(y_test, ypred)

"""The accuracy score of the model is 1.0."""

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, ypred)
print(confusion_matrix)

"""From the above result, we have 14,907 predicted results as True Positive and 0 predicted as False Positive.

While 0 predicted as False Negative and 8,971 predicted results as True Negative.

# **Step 9:Model Evaluation and Visualization**
"""

from sklearn.metrics import f1_score
f1_score(y_test, ypred, average='binary')

from sklearn import metrics
metrics.precision_score(y_test,ypred)

metrics.recall_score(y_test,ypred)

"""f1_score for my model is 1.0.  
precision_score for my model is 1.0.
recall_score for my model is 1.0.
"""

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, ypred)
sns.heatmap(cm, annot=True, cmap="Blues")

"""#**Step 10: Creating WebApp using Streamlit**"""