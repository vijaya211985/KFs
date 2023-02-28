
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns


df = pd.read_csv('https://github.com/KoonFoong/Hotel_Booking_Cancellation_Prediction/blob/main/hotel_bookings.csv')
d_copy = df



df2=df.drop(['lead_time','arrival_date_year','arrival_date_month','arrival_date_week_number','arrival_date_day_of_month',
            'stays_in_weekend_nights','stays_in_week_nights','is_repeated_guest','previous_cancellations','previous_bookings_not_canceled','agent','company','adr','total_of_special_requests','reservation_status','reservation_status_date'],axis=1)




df2=df2.dropna()

df2['children']=df2['children'].astype('int64')

df2['required_car_parking_spaces']=df2['required_car_parking_spaces'].astype('int64')

df2[df2.duplicated()]

df2.drop_duplicates()

df2.boxplot()

df2['adults'].plot(kind="box")

df2['babies'].plot(kind="box")


avg_adults = df2['adults'].mean()


std_adults = df2['adults'].std()


df2['Z_Score_Adults'] = (df2['adults'] - avg_adults)/std_adults


avg_children = df2['children'].mean()


std_children = df2['children'].std()


df2['Z_Score_Children'] = (df2['children'] - avg_children)/std_children


avg_babies = df2['babies'].mean()


std_babies = df2['babies'].std()


df2['Z_Score_Babies'] = (df2['babies'] - avg_babies)/std_babies


avg_required_car_parking_spaces = df2['required_car_parking_spaces'].mean()


std_required_car_parking_spaces = df2['required_car_parking_spaces'].std()


df2['Z_Score_Required_car_parking_spaces'] = (df2['required_car_parking_spaces'] - avg_required_car_parking_spaces)/std_required_car_parking_spaces


fig, axes = plt.subplots(2,2, figsize=(15,5))

sns.distplot(df2['adults'], ax=axes[0,0])
sns.distplot(df2['Z_Score_Adults'], ax=axes[0,1])
sns.distplot(df2['children'], ax=axes[1,0])
sns.distplot(df2['Z_Score_Children'], ax=axes[1,1])


plt.show()



fig, axes = plt.subplots(2,2, figsize=(15,5))

sns.distplot(df2['babies'], ax=axes[0,0])
sns.distplot(df2['Z_Score_Babies'], ax=axes[0,1])
sns.distplot(df2['required_car_parking_spaces'], ax=axes[1,0])
sns.distplot(df2['Z_Score_Required_car_parking_spaces'], ax=axes[1,1])


plt.show()



for i in df2.columns:
    if df2[i].dtypes=='object':
        df2[i] = pd.Categorical(pd.factorize(df2[i])[0])



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



a=d_copy['hotel'].value_counts()


ax = sns.countplot(x = d_copy['hotel']);

ax.set_ylabel("Total Bookings")


bx =sns.histplot(d_copy, x = 'is_canceled', hue = 'hotel', multiple = 'stack', palette = ['coral', 'skyblue']);

bx.set_ylabel("Total Bookings")



d_copy.groupby('is_canceled').size().plot(kind = 'pie', 
                                   autopct = '%.2f%%', 
                                   labels = ['Cancelled', 'Successful'], 
                                   label = 'Booking Status', 
                                   fontsize = 15,
                                   colors = ['red', 'green']);



plt.bar(d_copy['market_segment'], d_copy['previous_bookings_not_canceled'], width=0.7)
plt.title('Successful Bookings based on Market Segment')
plt.xlabel('Market Segment')
plt.ylabel('Previous Bookings Not Cancelled')
plt.show()



c=d_copy['reserved_room_type'].value_counts()


cx = sns.countplot(x = d_copy['reserved_room_type']);

cx.set_ylabel("Total Reservations")



!pip install sweetviz

import sweetviz as sv

hotel_report = sv.analyze(d_copy)

hotel_report.show_html('Hotel Booking Cancellation Prediction.html')
hotel_report.show_notebook()


from sklearn.model_selection import train_test_split

df3 = df2.drop(['adults','children','babies','required_car_parking_spaces'],axis=1)



df3.fillna(0, inplace=True)

X=pd.DataFrame(df3.iloc[:,0:25])# df.iloc[rows,cols]
y=pd.DataFrame(df3['is_canceled'])


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42,train_size=0.8)



from sklearn.linear_model import LogisticRegression
lm1= LogisticRegression()
lm1.fit(X_train, y_train)

lm1.coef_

lm1.intercept_

ypred=lm1.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, ypred)


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, ypred)
print(confusion_matrix)


from sklearn.metrics import f1_score
f1_score(y_test, ypred, average='binary')

from sklearn import metrics
metrics.precision_score(y_test,ypred)

metrics.recall_score(y_test,ypred)


from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, ypred)
sns.heatmap(cm, annot=True, cmap="Blues")
