# General Assembly Capstone Project

## Abstract: 

Sampling is a key component in the production of hip hop instrumentals. Sampling in this case refers to “the act of taking a portion, or sample, of one sound recording and reusing it as an instrument or a sound recording in a different song or piece” ([Wikipedia](https://en.wikipedia.org/wiki/Sampling_(music))). Manually listening to hundreds of audio segments in pursuit of a handful of useful ones is laborious, so I set out to determine *what features of a particular audio segment make it more or less sample-worthy than another audio segment* derived from the same source track. 

Since what defines a good sample is subjective, I listened to 3,196 audio segments [algorithmically sliced](https://www.image-line.com/support/FLHelp/html/plugins/Slicex.htm) from 4 songs and manually classified each segment as “sample-worthy” (1) or “not-sample-worthy” (0) according to my own preferences. I applied a number of digital signal transformations to the audio segment arrays to engineer potential features in a classification model. Through rigorous feature selection, I was able to return an XGBClassifier model with an ROC score of 0.68 and a recall score of 0.74. I then fed slices from an unseen song through the model, returning 32 segments ranked by the probability they would be classified as “sample-worthy.” Finally, I manually listened to the selected segments, further selected usuable sounds, and arranged those sounds to construct a new instrumental.

One such resulting instrumental can be heard [here](https://drive.google.com/open?id=0BxhJCrTr-R-BVGd3RGMzd2pRZms), with vocals and drums added for context.

## Data Acquisition

In order to make manually listening to and classifying a large number of audio segments effortless, I first had to create an application that would:
* feed me one looped segment
* wait for my keyboard input to classify it
* store that classification
* feed me the next segment

I accomplished this by utilizing the [Kivy](https://kivy.org/) library.

![alt text](https://github.com/dahmad/ga-capstone/blob/master/images/kivy.gif "Manually classifying segments through Kivy application")
*Figure 1.* Kivy application in action. See [code](https://github.com/dahmad/ga-capstone/blob/master/kivy-classification.py).

I paired the classifications with their corresponding audio segment filepaths in a CSV file. The individual CSV files for each souce track was then combined into one dataframe containing 3,196 rows.

## Data Transformations

Features were engineered by performing digital signal transformations from the [Librosa](https://github.com/librosa/librosa) library on each audio array then returning a variety of summary statistics on the transformed arrays, ensuring each value contained only a single number and not an array. (See [code](https://github.com/dahmad/ga-capstone/blob/master/transformation.py)). 

This is a kitchen-sink approach that required no prior knowledge of signal processing theory, which was necessary given time limitations. I had to trust that examining feature importances returned by a tree-based model would clarify which transformations were useful.

## Operationalizing the Outcome Variable
	
Given the skewed nature of the dataset (238 of 3,196 segments were classified as sample-worthy), I decided to select every row where “Classification” == 1 and then select an equal-sized random subsample of the rows where “Classification” == 0. The final truncated datatset contained 476 rows and 423 columns.

This step was necessary as any model that simply classified all test inputs as 0 would always be at least 92.55% accurate. Furthermore, the transformations required heavy processing power, and creating 400+ columns from 3000+ rows would be prohibitively time-consuming.

## Model Selection and Evaluation

A variety of classification algorithms were considered when running preliminary models. The XGBoost Classifier quickly proved to be consistently high-performing. 

The ROC score was selected as a scoring tool for its robustness in considering a variety of discrimination thresholds. Additionally, given that the cost of false positives is low in this problem, the recall score was also considered. Returning a handful of true positives amongst a couple dozen false positives is still preferable to the default: a handful of true positives in hundreds of unclassified segments. 

## Feature Selection

In order to select a manageable number of features, I first chose to iteratively look at how individual columns performed against the target. I ran each column against the target three times looking for average ROC scores that were above 0.60. This was necessary because ROC scores were varying wildly for the same column between different attempts depending on the specific train-test split. 

For the final model, I set x to the columns that averaged >0.60. I ran the script and pickled the resulting model, exporting that list of columns for use in the final script.

## Optimization

While the model was scoring well, it was still prohibitively time-consuming to return predictions from it since the signal transformations had to first be calculated before any predicting could occur. Each of the 22 columns took over a minute to form.

Looking at the feature importances of these columns, I saw a sharp drop after three and again after eight features. 

![alt text](https://github.com/dahmad/ga-capstone/blob/master/images/features.png "XGBClassifier Feature Importances")

*Figure 2.* XGBClassifier Feature Importances. See [code](https://github.com/dahmad/ga-capstone/blob/master/optimization.ipynb).

I ran models for both subsets of data and found that with only three features, I was able to actually increase recall at the expense of a few percentage points on the ROC score with the benefit of taking one eighth of the time to run. The model with three features became my final model.

## Final Result

I ran a list of segments from a brand new song through the final pickled model and ranked the segments by their predict_proba_ values. The top 32 segments were copied into a new folder. The selected segments were then uploaded into a digital audio workstation (FLStudio), where I was able to arrange five of those segments that I liked into a new instrumental. 

That resulting instrumental can be heard [here](https://drive.google.com/open?id=0BxhJCrTr-R-BVGd3RGMzd2pRZms), with existing vocals and drums added for context.

## Future Deployment Strategies

The biggest limiting factors were my lack of signal processing expertise and lack of heavy processing power. A larger dataset transformed in an informed way could potentially provide a more accurate model while continuing to minimize the number of features.

Furthermore, compiling the beat slicing, classification, and selection playback into one Kivy app would be a user-friendly way of deploying the model.