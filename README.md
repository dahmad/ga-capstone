# General Assembly Capstone Project

## Abstract: 

Sampling is a key component in the production of hip hop instrumentals. Sampling in this case refers to “the act of taking a portion, or sample, of one sound recording and reusing it as an instrument or a sound recording in a different song or piece” ([Wikipedia](https://en.wikipedia.org/wiki/Sampling_(music))). Manually listening to hundreds of audio segments in pursuit of a handful of useful ones is tedious and laborious, so I set out to determine what features of a particular audio segment make it more or less sample-worthy than another audio segment derived from the same source track. Since what defines a good sample is subjective, I listened to 3,196 audio segments [algorithmically sliced](https://www.image-line.com/support/FLHelp/html/plugins/Slicex.htm) from 4 songs and manually classified each segment as “sample-worthy” (1) or “not-sample-worthy” (0) according to my own preferences. I then applied a number of digital signal transformations to the audio segment arrays to be used as potential features in a classification model. Through rigorous feature selection, I was able to return an XGBClassifier model with an ROC score and a recall score above 0.70. I then fed slices from an unseen song through the model, returning 32 segments ranked by the probability they would be classified as “sample-worthy.” Finally, I was able to manually listen to the selected segments, further select usuable sounds, and then arrange those sounds to construct a new instrumental.

One resulting instrumental can be heard [here](https://drive.google.com/open?id=0BxhJCrTr-R-BVGd3RGMzd2pRZms), with vocals and drums added for context.

## Data Acquisition

In order to make manually listening to and classifying a large number of audio segments effortless, I first had to create an application that would
* feed me one segment
* wait for my keyboard input to classify it
* store that classification
* then feed me the next segment. 

I accomplished this by utilizing the [Kivy](https://kivy.org/) library.

![alt text](https://github.com/dahmad/ga-capstone/blob/master/images/kivy.gif "Manually classifying segments through Kivy application")

I then paired the classifications to the corresponding audio segment filepaths in a dataframe that could be exported to a CSV file. The individual CSV files for each souce track was then combined into one dataframe containing 3,196 rows.

## Data Transformations

Features were engineered by performing digital signal transformations from the [Librosa](https://github.com/librosa/librosa) library on each audio array then returning a variety of summary statistics on the transformed arrays. This is a kitchen-sink approach that required no prior knowledge of signal processing theory, which was necessary given time limitations. The primary goals were to have a single number contained in each value as opposed to an array and to trust that examining feature importances returned by a tree-based model would clarify which transformations were useful.

## Operationalizing the Outcome Variable
	
Given the skewed nature of the dataset (238 of 3,196 rows were classified as 1), I decided to select every row where “Classification” == 1 and then select an equal-sized random subsample of the rows where “Classification” == 0. The final truncated datatset contained 476 rows and 423 columns.

This step was necessary as any model that simply classified all test inputs as 0 would always be at least 92.55% accurate. Furthermore, the transformations required heavy processing power, and creating 400+ columns from 3000+ rows would be prohibitively time-consuming.

## Model Selection and Evaluation

A variety of classification algorithms were considered when running preliminary models. The XGBoost Classifier and AdaBoost Classifier quickly proved to be consistently high-performing. Additionally, a soft-voting classifier utilizing these two models was included.

The ROC score was selected as a scoring tool for its robustness in considering a variety of discrimination thresholds. However, given that the cost of false positives is low in this problem, the recall score was also considered. Returning a handful of true positives amongst a couple dozen false positives is still preferable to a handful of true positives in hundreds of unclassified segments. 

Upon realization that calling feature importances from tree-based models was an ineffective method of feature selection, I chose to iteratively look at how individual columns performed against the target. For each column, I set x equal to that column, ran a given model three times with three separate train-test splits, and then averaged the three resulting ROC scores. This was necessary because ROC scores were varying wildly for the same column between different attempts depending on the specific train-test split. Any column that returned an average ROC score greater than 0.60 was retained for the final models. This process was repeated three times for each of the three models.

For each model, a final train-test split was performed and the model was fit with the dataframe subset containing only the returned columns. The ROC and recall scores were printed for each of the three models to be compared against one another. With each successive running of the script, it became clear that the XGBoost Classifier was reliably high-performing. I pickled an XGBoost Classifier model that returned an ROC and recall score above 0.70.

I then ran a list of segments from a brand new song through this pickled model and ranked the segmetns by their predict_proba_ values. The top 32 segments were copied into a new folder. The selected segments were then uploaded into a digital audio workstation (FLStudio). I then arranged five segments that I liked into a new instrumental. 

That resulting instrumental can be heard [here](https://drive.google.com/open?id=0BxhJCrTr-R-BVGd3RGMzd2pRZms), wubith existing vocals and drums added for context.

## Future Deployment Strategies

The biggest limiting factors were my lack of signal processing expertise and lack of heavy processing power. A larger dataset transformed in an informed way could potentially provide a more accurate model with fewer features, which would in turn require less time. Each of the 22 columns in the final model takes approximately one minute to create before the model can begin classifying the new inputs.

Furthermore, compiling the beat slicing, classification, and selection playback into one Kivy app would be a user-friendly way of deploying the model.