import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from loadData import combine_data, select_features, normalize_features
import numpy as np
import seaborn as sns
import pprint as pp
FeaturesByDataSet = {
'AllFeatures': ['accouncementcount', 'discussioncount', 'groupdiscussioncount', 'individualdiscussioncount',
                      'sectioncount', 'assignmentcount', 'gradedassignmentcount', 'modulecount', 'wikipagecount',
                      'attachementcount', 'groupcount', 'quizcount', 'quizquestioncount', 'externaltoolcount',
                      'moduleurlcount', 'tacount', 'studentcount', 'LengthInWeeks','LOCATION_Online', 'LOCATION_OnCampus'],
'wo_StdntCnt_Loc':['accouncementcount', 'discussioncount', 'groupdiscussioncount', 'individualdiscussioncount',
                      'sectioncount', 'assignmentcount', 'gradedassignmentcount', 'modulecount', 'wikipagecount',
                      'attachementcount', 'groupcount', 'quizcount', 'quizquestioncount', 'externaltoolcount',
                      'moduleurlcount', 'tacount', 'LengthInWeeks'],
'wo_StdntCnt_Loc_Len':['accouncementcount', 'discussioncount', 'groupdiscussioncount', 'individualdiscussioncount',
                      'sectioncount', 'assignmentcount', 'gradedassignmentcount', 'modulecount', 'wikipagecount',
                      'attachementcount', 'groupcount', 'quizcount', 'quizquestioncount', 'externaltoolcount',
                      'moduleurlcount', 'tacount'],
'wo_StdntCnt_Loc_Len_GD':['accouncementcount', 'groupdiscussioncount', 'individualdiscussioncount',
                      'sectioncount', 'gradedassignmentcount', 'modulecount', 'wikipagecount',
                      'attachementcount', 'groupcount', 'quizcount', 'quizquestioncount', 'externaltoolcount',
                       'tacount','moduleurlcount'],
'wo_StdntCnt_Loc_Len_url':['accouncementcount', 'discussioncount', 'groupdiscussioncount', 'individualdiscussioncount',
                      'sectioncount', 'assignmentcount', 'gradedassignmentcount', 'modulecount', 'wikipagecount',
                      'attachementcount', 'groupcount', 'quizcount', 'quizquestioncount', 'externaltoolcount',
                       'tacount']

}


CountFeatures = pd.read_csv(r".\input\CountFeatures.csv")
Metadata = pd.read_csv(r".\input\Metadata.csv")
scale_type = 'Standard'
print(CountFeatures.groupby('studentcount')['code'].count())
combined_df = combine_data(CountFeatures, Metadata)
Metadata = Metadata[Metadata["SERVICE_ID"] == 'Canvas']

combined_df = combined_df[(combined_df['studentcount'] > 0) & (combined_df['studentcount'] <= 700)].reset_index()


columns_to_plot = ['accouncementcount', 'discussioncount', 'groupdiscussioncount', 'individualdiscussioncount',
                      'sectioncount', 'assignmentcount', 'gradedassignmentcount', 'modulecount', 'wikipagecount',
                      'attachementcount', 'groupcount', 'quizcount', 'quizquestioncount', 'externaltoolcount',
                      'moduleurlcount']
pp.pprint(combined_df.describe())
data = normalize_features(combined_df, columns_to_plot, scale_type)

data = select_features(combined_df, columns_to_plot)
data.replace([np.inf, -np.inf], np.nan).dropna( how="all")
pp.pprint(data.shape)
pp.pprint(data.describe())
sns.set(style="ticks", color_codes=True)
sns.pairplot(data)
plt.show()
