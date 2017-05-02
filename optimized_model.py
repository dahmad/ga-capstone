import numpy as np
import time
from sklearn.externals import joblib
from glob import glob
import pandas as pd
import librosa
import scipy.stats as sps
from xgboost import XGBClassifier
import shutil

# Seeing how long transformations take
start_time = time.time()

# Loading pickled model
model = joblib.load('xgboost_three.pkl') 

# Compiling filepaths for segments to be classified
files = glob('assets/lafoule/*.wav')
df = pd.DataFrame(files, columns=['filepath'])

# Tranforming individual columns
df['ifgram_std_cov_kurtosis'] = df['filepath'].apply(lambda x: np.std(np.cov(sps.describe(librosa.core.ifgram(librosa.core.load(x)[0])).kurtosis)))
df['tonnetz_min_variance'] = df['filepath'].apply(lambda x: np.min(sps.describe(librosa.feature.tonnetz(librosa.core.load(x)[0])).variance))
df['chroma_cqt_mean_minmax'] = df['filepath'].apply(lambda x: np.mean(sps.describe(librosa.feature.chroma_cqt(librosa.core.load(x)[0])).minmax))

# Operationalizing data
x = df.drop(['filepath'], axis=1)
columns = pd.DataFrame(x.columns, columns=['columns'])

# Getting ranked predict_proba
df = df.join(pd.DataFrame(model.predict_proba(x), columns=['0','1']))
proba_sort = df.sort_values('1', ascending=False)
ranked_files = proba_sort['filepath'].values

# Exporting to file
df.to_csv('df.csv')

# Copying top 32 files to new folder
for x in range(32):
    shutil.copy2(ranked_files[x], 'assets/lafoule/selections/%s.wav' % '{0:03d}'.format(x + 1))