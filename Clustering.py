import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from loadData import combine_data, normalize_features,select_features

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

def plot_sum_squared_distances(train_df, max_clusters,scale_type, datasetname):
    sum_of_squared_distances=[]
    range_k = range(1, max_clusters)
    for k in range_k:
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=200, n_init=15)
        kmeans.fit(train_df)
        print("squared error for iteration " + str(k) + " is: " + str(kmeans.inertia_))
        sum_of_squared_distances.append(kmeans.inertia_)

    plt.figure(1)
    plt.plot(range_k, sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    # plt.show()
    plt.savefig("./Output/sum_squared_distances_"+scale_type+"_"+datasetname+".png")
    plt.close()

CountFeatures = pd.read_csv(r".\input\CountFeatures.csv")
Metadata = pd.read_csv(r".\input\Metadata.csv")
scale_type = 'Standard'
print(CountFeatures.groupby('studentcount')['code'].count())
combined_df = combine_data(CountFeatures, Metadata)
Metadata = Metadata[Metadata["_ID"] == 'Canvas']
combined_df = combined_df[~(combined_df['coursename'].str.contains("asu", case=False) & combined_df['coursename'].str.contains("orientation", case=False))]
combined_df = combined_df[(combined_df['studentcount'] > 0) & (combined_df['studentcount'] <= 700)].reset_index()


columns_to_normalize = ['accouncementcount', 'discussioncount', 'groupdiscussioncount', 'individualdiscussioncount',
                      'sectioncount', 'assignmentcount', 'gradedassignmentcount', 'modulecount', 'wikipagecount',
                      'attachementcount', 'groupcount', 'quizcount', 'quizquestioncount', 'externaltoolcount',
                      'moduleurlcount', 'tacount', 'studentcount', 'LengthInWeeks','LOCATION_Online', 'LOCATION_OnCampus']
train_df = normalize_features(combined_df, columns_to_normalize, scale_type)


for dataset in FeaturesByDataSet.keys():
    train_df = normalize_features(combined_df, columns_to_normalize, scale_type)
    print("Processing for data set : "+ dataset)
    features_for_clustering = FeaturesByDataSet[dataset]
    train_df = select_features(train_df, features_for_clustering)
    print(combined_df.shape, train_df.shape)

    plot_sum_squared_distances(train_df,30,scale_type,dataset)
    NoOfClusters = 7

    kmeans = KMeans(n_clusters=NoOfClusters, init='k-means++', max_iter=200, n_init=15)
    kmeans.fit(train_df)

    labels_ = kmeans.labels_
    combined_df['clusters_sklearn_kmeans'] = labels_

    # plt.cla()
    # plt.clf()
    plt.figure(2)
    plt.title('HCD of Course with dataset:' + str(dataset))
    plt.xlabel('Courses')
    plt.ylabel('Clusters')
    dendrogram = sch.dendrogram(sch.linkage(train_df, method='ward'))
    plt.savefig('./Output/HCD_' + str(dataset) + '.png', dpi=100)
    plt.close()
    hc = AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
                            connectivity=None, linkage='ward', memory=None, n_clusters=NoOfClusters,
                            pooling_func='deprecated')

    y_hc = hc.fit_predict(train_df)
    labels_ = y_hc
    combined_df['clusters_sklearn_hc'] = labels_

    labelled_data = combined_df.to_csv("./Output/lbl_"+dataset+".csv", sep=',', encoding='utf-8')

    combined_df.head(3)
    print(combined_df.groupby('clusters_sklearn_hc')['code'].count())
    print(combined_df.groupby('clusters_sklearn_kmeans')['code'].count())



