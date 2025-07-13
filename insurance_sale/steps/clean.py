import numpy as np
from sklearn.impute import SimpleImputer

class cleaner:
    def __init__(self):
        # Initialize the SimpleImputer with the strategy to fill missing values
        # Here, we use 'most_frequent' to fill NaN values with the most frequent value in each column
        self.imputer= SimpleImputer(strategy= 'most_frequent', missing_values= np.nan)

    def clean_data(self, data):
        data.drop(['id', 'SalesChannelID', 'VehicleAge', 'DaysSinceCreated'], axis=1, inplace=True)
        data ['AnnualPremium'] = data['AnnualPremium'].str.replace(',', '').astype(float)
        for col in ['Gender', 'RegionID']:
            data[col]  =self.imputer.fit_transform(data[[col]]).flatten()

        data['Age'] = data['Age'].fillna(data['Age'].median())
        data['HasDrivingLicense'] = data['HasDrivingLicense'].fillna(1)
        data['Switch'] = data['Switch'].fillna(-1)
        data['PastAccident'] = data['PastAccident'].fillna("Unknown", inplace=False)

# stats formula
        Q1 = data['AnnualPremium'].quantile(0.25)
        Q3 = data['AnnualPremium'].quantile(0.75)
        IQR = Q3-Q1
        upper_bound= Q3+ 1.5*IQR
        data= data[data['AnnualPremium'] <= upper_bound]

        return data