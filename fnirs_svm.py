from Dataset.Prep import fnirs_utils
import numpy as np
from sklearn.model_selection import train_test_split

#load the fnirs data
fnirs_data = fnirs_utils.load_fnirs("./Dataset/Data/fNIRS")

#initialize the lists to store the data
x = []
y = []
subjectIds = []

#iterate over the data
for data in fnirs_data[0]:
   
    # get the label
    label = int(data["pheno"]["nback"])
   
  
    x.append(data["roiTimeseries"].T)
    y.append(label)
    subjectIds.append(int(data["pheno"]["subjectId"][1]))


# make the data the proper shape [conditions, subjects, trials, channels, timepoints]
x = np.array(x)
y = np.array(y)
subjectIds = np.array(subjectIds)
# Define the number of conditions, subjects, trials, channels, and timepoints
n_conditions = len(np.unique(y))  # Number of unique conditions (e.g., 2 for binary)
n_subjects = len(np.unique(subjectIds))  # Number of subjects
n_trials = x.shape[0] // (n_conditions * n_subjects)  # Trials per condition per subject
n_channels = x.shape[1]  # Number of fNIRS channels
n_timepoints = x.shape[2]  # Number of time points

# Reshape x into the desired format: [conditions, subjects, trials, channels, timepoints]
x_reshaped = x.reshape((n_conditions, n_subjects, n_trials, n_channels, n_timepoints))

# Verify alignment of labels y
# Ensure y is shaped: [conditions, subjects, trials]
y_reshaped = y.reshape((n_conditions, n_subjects, n_trials))

# Subject IDs should match: [subjects]
subjectIds_reshaped = subjectIds[:n_subjects]

print("Reshaped x shape:", x_reshaped.shape)
print("Reshaped y shape:", y_reshaped.shape)
print("Reshaped subjectIds shape:", subjectIds_reshaped.shape)


# #reshape the data
# print("Before reshaping: ", x.shape)
# x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
# print("After reshaping: ", x.shape)



# #split the data into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# #initialize the SVM model
# from sklearn.svm import SVC
# model = SVC(kernel='linear')

# #train the model
# model.fit(x_train, y_train)

# #test the model
# y_pred = model.predict(x_test)

# #calculate the accuracy
# from sklearn.metrics import accuracy_score
# from scipy.spatial.distance import pdist, squareform

# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy: ", accuracy)

from neurora.rdm_cal import eegRDM
from neurora.rsa_plot import plot_rdm, plot_corrs_by_time, plot_nps_hotmap, plot_corrs_hotmap
# Append the label as the first dimension of x

# Choose a metric: 'correlation', 'euclidean', etc.
rdm = eegRDM(x_reshaped)
plot_rdm(rdm, percentile=True, title="fNIRS RDM") 