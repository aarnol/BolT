from Dataset.Prep import fnirs_utils
import numpy as np
from sklearn.model_selection import train_test_split

#load the fnirs data
fnirs_data = fnirs_utils.load("./Dataset/Data/fNIRS/Preprocessed/", "HbR")

#initialize the lists to store the data
x = []
y = []
subjectIds = []

#iterate over the data
for data in fnirs_data:
   
    # get the label
    label = int(data["pheno"]["label"])
   
    
    #create the feature vector
    data["roiTimeseries"] = data['roiTimeseries'].T
    
    # roiTimeseries is a 2D array with shape (n_channels, n_timepoints)
    #extract mean, std, range, kurtosis, skewness, and entropy for each channel
    #mean
    mean = np.mean(data["roiTimeseries"], axis=1)
    #std
    std = np.std(data["roiTimeseries"], axis=1)
    #range
    range_ = np.max(data["roiTimeseries"], axis=1) - np.min(data["roiTimeseries"], axis=1)
    # kurtosis
    kurtosis = np.apply_along_axis(lambda x: np.mean((x - np.mean(x))**4) / (np.std(x)**4), axis=1, arr=data["roiTimeseries"])
    # skewness
    skewness = np.apply_along_axis(lambda x: np.mean((x - np.mean(x))**3) / (np.std(x)**3), axis=1, arr=data["roiTimeseries"])
    # entropy
    # entropy = np.apply_along_axis(lambda x: -np.sum((x / np.sum(x)) * np.log2(x / np.sum(x) + 1e-9)), axis=1, arr=data["roiTimeseries"])
    #check if any have nan values and say which one
    if np.isnan(mean).any():
        print("Mean has nan values")
    if np.isnan(std).any():
        print("Std has nan values")
    if np.isnan(range_).any():
        print("Range has nan values")
    if np.isnan(kurtosis).any():
        print("Kurtosis has nan values")
    if np.isnan(skewness).any():
        print("Skewness has nan values")
    
    # concatenate the features into a single array
    features = np.concatenate((mean, std, range_, kurtosis, skewness), axis=0)
    
    # append the features to the list
    x.append(features)
    y.append(label)
    subjectIds.append(int(data["pheno"]["subjectId"][1]))



#convert the lists to numpy arrays
x = np.array(x)
y = np.array(y)



#reshape the data
print("Before reshaping: ", x.shape)
# x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
# print("After reshaping: ", x.shape)



#split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#initialize the SVM model
from sklearn.svm import SVC
model = SVC(kernel='poly', degree=3, C=1.0, gamma='scale', class_weight='balanced')

#train the model
model.fit(x_train, y_train)

#test the model
y_pred = model.predict(x_test)

#calculate the accuracy
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import pdist, squareform

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

