#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% Libraries imported
import random
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
from sklearn import tree

#%% Preprocessing

# sets random seed
Nnumber = 100
np.random.seed(seed = Nnumber)
random.seed(Nnumber)

#importing the data
df = pd.read_csv('spotify52kData.csv')

#importing the data into numpy for easier calculation 
data = df.to_numpy()

# creating individual arrays for certain song features
durationArr = data[:,5 ].astype(int)
danceabilityArr = data[:,7 ].astype(float)
energyArr = data[:,8].astype(float)
loudnessArr = data[:, 10].astype(float)
speechinessArr = data[:, 12].astype(float)
acousticnessArr = data[:,13 ].astype(float)
instrumentalnessArr = data[:, 14].astype(float)
livenessArr = data[:,15 ].astype(float)
valenceArr = data[:,16 ].astype(float)
tempoArr = data[:,17 ].astype(float)

genreArr = data[:,19 ]

# combining the ten song features into one matrix so its easier to work with
allTenFeaturesDf = pd.concat([df["duration"], df["danceability"], df["energy"], df["loudness"], df["speechiness"], df["acousticness"],df["instrumentalness"], df["liveness"], df["valence"], df["tempo"] ], axis =1)
allTenFeaturesMatrix = allTenFeaturesDf.to_numpy()
#%% Question 1
fig, axs = plt.subplots(2, 5, figsize=(6, 3), gridspec_kw={'wspace': 1.2, 'hspace': 0.5},)



axs[0,0].hist(durationArr, bins = 25)
axs[0,0].title.set_text("Duration")
axs[0,0].ticklabel_format(axis = 'x', style = 'plain')
axs[0,1].hist(danceabilityArr)
axs[0,1].title.set_text("Danceability")
axs[0,2].hist(energyArr)
axs[0,2].title.set_text("Energy")
axs[0,3].hist(loudnessArr)
axs[0,3].title.set_text("Loudness")
axs[0,4].hist(speechinessArr)
axs[0,4].title.set_text("Speechiness")
axs[1,0].hist(acousticnessArr)
axs[1,0].title.set_text("Acousticness  ")
axs[1,1].hist(instrumentalnessArr)
axs[1,1].title.set_text("Instrumental")
axs[1,2].hist(livenessArr)
axs[1,2].title.set_text("Liveness")
axs[1,3].hist(valenceArr)
axs[1,3].title.set_text("Valence")
axs[1,4].hist(tempoArr)
axs[1,4].title.set_text("Tempo")
axs[1,4].ticklabel_format(axis = 'x', style = 'plain')


plt.hist(durationArr, bins = 150)
plt.title("Duration")

#%% Question 2
popularityArr = data[:,4].astype(int)

songLengthPopularCorr = np.corrcoef(durationArr, popularityArr)
plt.scatter(durationArr, popularityArr)
plt.ticklabel_format(axis = 'x', style = 'plain')
plt.title('r = {:.3f}'.format(songLengthPopularCorr[0,1]))
plt.xlabel("Duration/Song Length in milliseconds")
plt.ylabel("Popularity(0-100)")

#%% Question 3

#extracting the subset of popularity and explicit columns from the dataframe and concatenating them
explicitdf = df["explicit"]
popularitydf = df["popularity"]
explicitAndPopularityDf = pd.concat([explicitdf, popularitydf], axis = 1)

# this divides the subset of popularity vs explicit data into 2 groups: explicit and non explicit
explicitVsPopularityDf = explicitAndPopularityDf[explicitAndPopularityDf["explicit"]==True]
NonexplicitVsPopularityDf = explicitAndPopularityDf[explicitAndPopularityDf["explicit"]==False]

# this converts the pandas dataframes into numpy arrays so it is easier to work with for the hypothesis testing
explicitVsPopularityArr = explicitVsPopularityDf.to_numpy().astype(int)
NonexplicitVsPopularityArr = NonexplicitVsPopularityDf.to_numpy().astype(int)

# First, I plotted a histogram to see the distribution and shape of the data and found the medians 
plt.hist(NonexplicitVsPopularityArr[:,1])
plt.title("Not explicit and Popularity     Median = {:.3f}".format(np.median(NonexplicitVsPopularityArr[:,1])))
plt.xlabel("Popularity(0-100)")
plt.ylabel("Count (number of songs)")

plt.hist(explicitVsPopularityArr[:,1])
plt.title("Explicitly Rated and Popularity     Median = {:.3f}".format(np.median(explicitVsPopularityArr[:,1])))
plt.xlabel("Popularity(0-100)")
plt.ylabel("Count (number of songs)")

# Mann-Whitney U test
u1,p1 = stats.mannwhitneyu(NonexplicitVsPopularityArr[:,1], explicitVsPopularityArr[:,1])
print(p1)


#%% Question 4

#extracting the subset of popularity and mode columns from the dataframe and concatenating them
modeMajMinDf = df["mode"]
modeAndPopularityDf = pd.concat([modeMajMinDf, popularitydf], axis = 1)

# this divides the subset of popularity vs mode data into 2 groups: major and minor
majorVsPopularityDf = modeAndPopularityDf[modeAndPopularityDf["mode"]==1]
minorVsPopularityDf = modeAndPopularityDf[modeAndPopularityDf["mode"]==0]

# this converts the pandas dataframes into numpy arrays so it is easier to work with for the hypothesis testing
majorVsPopularityArr = majorVsPopularityDf.to_numpy().astype(int)
minorVsPopularityArr = minorVsPopularityDf.to_numpy().astype(int)

# Then, I plotted a histogram to see the distribution and shape of the data and found the medians 
plt.hist(majorVsPopularityArr[:,1])
plt.title("Major Key and Popularity     Median = {:.3f}".format(np.median(majorVsPopularityArr[:,1])))
plt.xlabel("Popularity(0-100)")
plt.ylabel("Count (number of songs)")

plt.hist(minorVsPopularityArr[:,1])
plt.title("Minor Key and Popularity      Median = {:.3f}".format(np.median(minorVsPopularityArr[:,1])))
plt.xlabel("Popularity(0-100)")
plt.ylabel("Count (number of songs)")

# Mann-Whitney U test
u2,p2= stats.mannwhitneyu(majorVsPopularityArr[:,1], minorVsPopularityArr[:,1])
print(p2)

#%% Question 5
#first I calculated the spearman's rank coefficient
loudnessEnergyRho = stats.spearmanr(loudnessArr, energyArr)

# then I created a scatterplot to visualize the data
plt.scatter(loudnessArr, energyArr)
plt.title("$\\rho$ = {:.3f}".format(loudnessEnergyRho.statistic))
plt.xlabel("Average loudness in decibels")
plt.ylabel("Energy(0-1)")

#%% Question 6

# This creates a 10X3 matrix to hold data for each of the 10 song features
# the first column is the slope, second columns is the y-intercept, and the third column is the R^2
bestPredictorMatrix = np.empty([10,3])
yPredicted = np.empty([len(allTenFeaturesMatrix), 10])
for i in range (10):
    x = allTenFeaturesMatrix[:,i].reshape(len(allTenFeaturesMatrix), 1)
    RegressionModel = LinearRegression().fit(x,popularityArr)
    bestPredictorMatrix[i, 0] = RegressionModel.coef_
    bestPredictorMatrix[i, 1] = RegressionModel.intercept_
    bestPredictorMatrix[i, 2] = RegressionModel.score(x,popularityArr)
    yPredicted[:, i] = allTenFeaturesMatrix[:,i] * bestPredictorMatrix[i, 0] + bestPredictorMatrix[i,1]

# plotting scatterplots for the all of the song features versus popularity to observe the shape
TenfeaturesName = np.array(["duration", "danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"])
for i in range (10):
    plt.scatter(allTenFeaturesMatrix[:,i],popularityArr,marker ='o') 
    plt.xlabel(TenfeaturesName[i]) 
    plt.ylabel('popularity')  
    plt.scatter(allTenFeaturesMatrix[:,i],yPredicted[:,i],marker ='o',color='black')
    plt.show()

#%% Question 7

multipleRegressionModel = LinearRegression().fit(allTenFeaturesMatrix, popularityArr)
rSquaredMultipleRegression = multipleRegressionModel.score(allTenFeaturesMatrix, popularityArr)
popularityPredictedMultipleRegression = multipleRegressionModel.predict(allTenFeaturesMatrix)

plt.scatter(popularityPredictedMultipleRegression, popularityArr, marker ='o') 
plt.xlabel('popularity predicted') 
plt.ylabel('popularity actual')  
plt.title("R^2 : {:.3f}".format(rSquaredMultipleRegression))
plt.show()


#%% Question 8

#First I looked at a heat map of the correlations to get a general idea of which features are correlated
featuresCorrMatrix = np.corrcoef(allTenFeaturesMatrix, rowvar= False)
plt.imshow(featuresCorrMatrix)
plt.xlabel("features")
plt.ylabel("features")
plt.colorbar()
plt.show()

# Doing the PCA
zscoredFeatures = stats.zscore(allTenFeaturesMatrix)
pca = PCA().fit(zscoredFeatures)
eigVals = pca.explained_variance_
loadings = pca.components_
rotatedData = pca.fit_transform(zscoredFeatures)

#scree plot
#using the kaiser criterion, there are 3 principal components 
features = np.linspace(1, len(allTenFeaturesMatrix[1,:]), len(allTenFeaturesMatrix[1,:]))
plt.bar(features, eigVals)
plt.plot([0,len(allTenFeaturesMatrix[1,:])],[1,1],color='orange')
plt.xlabel("principal component")        
plt.ylabel("eigenvalue")       
plt.show() 


#looking at loadings data

#first principal component 
plt.bar(features, loadings[0, :]*-1)
plt.xlabel("Feature")
plt.ylabel("Loadings First Principal Component")
plt.show()
# The highest bars is feature 3 and 4, which corresponds to energy and loudness

#second principal component  
plt.bar(features, loadings[1, :]*-1)
plt.xlabel("Feature")
plt.ylabel("Loadings Second Principal Component")
plt.show()
# The highest bars is feature 2 and 9, which corresponds to danceability and valence

#third principal component 
plt.bar(features, loadings[2, :]*-1)
plt.xlabel("Feature")
plt.ylabel("Loadings Third Principal Component")
plt.show()
# The highest bars in the negative direction are feature 5 and 8, which corresponds to low/less speechiness and low/less liviness 



#variance explained
varExplained = eigVals/sum(eigVals)*100
varExplainedBy3PC = varExplained[0] + varExplained[1]+ varExplained[2]
print(varExplainedBy3PC)

# 2D graphs comparing the rotated data for 2 principle components at once
plt.plot(rotatedData[:,0]*-1, rotatedData[:,1]*-1, 'o', markersize = 5)
plt.xlabel("energy and loudness")
plt.ylabel("danceability and valence")
plt.show()

plt.plot(rotatedData[:,0]*-1, rotatedData[:,2]*-1, 'o', markersize = 5)
plt.xlabel("energy and loudness")
plt.ylabel("low/less speechiness and low/less liviness")
plt.show()

plt.plot(rotatedData[:,1]*-1, rotatedData[:,2]*-1, 'o', markersize = 5)
plt.xlabel("danceability and valence")
plt.ylabel("low/less speechiness and low/less liviness")
plt.show()


#3D graph of rotated data for all 3 principle components
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel("energy and loudness", fontsize = 7)
ax.set_ylabel("danceability and valence", fontsize = 7)
ax.set_zlabel("low/less speechiness and low/less liviness", fontsize = 7)
ax.zaxis.labelpad= -3
xAxis = rotatedData[:,0]*-1
yAxis = rotatedData[:,1] * -1
zAxis = rotatedData[:,2] * -1

ax.scatter(xAxis ,yAxis ,zAxis )


#%% Question 9

x = valenceArr.reshape(len(valenceArr), 1)
y = modeMajMinDf.to_numpy()

plt.scatter(valenceArr, y, color = 'black')
plt.xlabel("valence")
plt.ylabel("Major or Minor")

LogisticRegressionModel = LogisticRegression().fit(x,y)
x1 = np.linspace(0,1)
y1 = x1 * LogisticRegressionModel.coef_ + LogisticRegressionModel.intercept_
sigmoid = expit(y1)

plt.plot(x1, sigmoid.ravel(), color = 'red')
plt.scatter(valenceArr, y, color = 'black')
plt.xlabel("valence")
plt.ylabel("Major or Minor")

AUCscoreLogisticRegression = roc_auc_score(y, LogisticRegressionModel.decision_function(x))
print(AUCscoreLogisticRegression)

# plotting scatterplots for song features versus major/minor keys 
for i in range (10):
    plt.scatter(allTenFeaturesMatrix[:,i],y,marker ='o') 
    plt.xlabel(TenfeaturesName[i]) 
    plt.ylabel('Major or Minor')  
    plt.show()
#%% Question 10

x_train, x_test, y_train, y_test = train_test_split(allTenFeaturesMatrix, genreArr, random_state=Nnumber, test_size=0.3)

# train the decision tree with training data
decisionTree = DecisionTreeClassifier(random_state = Nnumber)
decisionTree.fit(x_train, y_train)

# test the accuracy using the testing data
DecisionTreePredictions = decisionTree.predict(x_test)
DecisionTreeAccuracy = accuracy_score(y_test, DecisionTreePredictions)
print(DecisionTreeAccuracy)

tree.plot_tree(decisionTree, max_depth = 1, fontsize= 5)

#%% Extra Credit
alt_rockSongs = df[df["track_genre"] == "alt-rock"]
alt_rockPopularityArr = alt_rockSongs["popularity"].to_numpy()
ExplicitAltRock = pd.concat([alt_rockSongs["explicit"], alt_rockSongs["popularity"]], axis = 1)
ExplicitArr = ExplicitAltRock[ExplicitAltRock["explicit"]==True].to_numpy().astype(int)
NonExplicitArr = ExplicitAltRock[ExplicitAltRock["explicit"]==False].to_numpy().astype(int)




plt.hist(ExplicitArr[:,1])
plt.title("Explicit Songs and Popularity     Median = {:.3f}".format(np.median(ExplicitArr[:,1])))
plt.xlabel("Popularity(0-100)")
plt.ylabel("Count (number of songs)")

plt.hist(NonExplicitArr[:,1])
plt.title("Non-explicit Songs and Popularity      Median = {:.3f}".format(np.median(NonExplicitArr[:,1])))
plt.xlabel("Popularity(0-100)")
plt.ylabel("Count (number of songs)")

# Mann-Whitney U test
u3,p3= stats.mannwhitneyu(ExplicitArr[:,1], NonExplicitArr[:,1])
print(p3)
