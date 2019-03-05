import pandas as pd
import numpy as np
# from sklearn.preprocessing import MinMaxScaler,StandardScaler

def label_location(row):
   if row['LOCATION'] in ["ONLINE","ICOURSE"] :
      return 'Online'
   else:
       return 'OnCampus'


def combine_data(CountFeatures, Metadata):
#   Modified and renamed fields to hide sensitive data
    Metadata = Metadata[Metadata['SERVICE_ID']=='Canvas'].reset_index()
    Metadata['START_DT'] = pd.to_datetime(Metadata['START_DT'])
    Metadata['END_DT'] = pd.to_datetime(Metadata['END_DT'])
    Metadata['LengthInWeeks'] = ((Metadata['END_DT'] - Metadata['START_DT']) / np.timedelta64(1, 'W')).round()
    Metadata = Metadata[Metadata['LengthInWeeks']!=0].reset_index()

    CountFeatures['code'] = CountFeatures['code'].str.lower()
    combined_df = pd.merge(CountFeatures, Metadata,
                           how='inner',
                           left_on=['code'], right_on=['code'])
    combined_df['LOCATION'] = combined_df.apply(lambda row: label_location(row), axis=1)
    categorical_features = ['LOCATION']

    for col in categorical_features:
        dummies = pd.get_dummies(combined_df[col], prefix=col)
        combined_df = pd.concat([combined_df, dummies], axis=1)
        combined_df.drop(col, axis=1, inplace=True)

    combined_df.fillna(0, inplace=True)
    for i, column in enumerate(combined_df):
        if column in ['accouncementcount', 'discussioncount','groupdiscussioncount', 'individualdiscussioncount',
                          'assignmentcount', 'gradedassignmentcount', 'modulecount', 'wikipagecount',
                          'attachementcount', 'quizcount', 'quizquestioncount',
                          'moduleurlcount']:
            combined_df[column] = combined_df[column] / combined_df['LengthInWeeks']

    combined_df.fillna(0, inplace=True)

    return combined_df


def normalize_features(data, columns, normalizer):
    temp = data.copy(deep=True)
    temp = temp[columns]

    if normalizer == 'MinMax':
        # mms = MinMaxScaler()
        # mms.fit(temp)
        # return mms.transform(temp)
        for i, column in enumerate(temp):
            if column in columns:
                temp[column] = (temp[column] - temp[column].min()) / (temp[column].max() - temp[column].min())
        temp.fillna(0, inplace=True)
    else:
        for i, column in enumerate(temp):
            if column in columns:
                temp[column] = (temp[column] - temp[column].mean()) / temp[column].std()
        temp.fillna(0, inplace=True)
    return temp


def select_features(data, project_columns):
    temp = data.copy(deep=True)
    return temp[project_columns]
