from glob import glob
import librosa
import pandas as pd
from sklearn.externals import joblib
import scipy.stats as sps
from xgboost import XGBClassifier
import numpy as np
import time
from multiprocessing import Pool
import shutil

# Seeing how long transformations take
start_time = time.time()

# Loading pickled model
model = joblib.load('xgboost.pkl') 

# Compiling filepaths for segments to be classified
files = glob('assets/foreigner/*.wav')
df = pd.DataFrame(files, columns=['filepath'])

# Parallelizing for performance
num_partitions = 12
num_cores = 4

def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

# Tranforming individual columns
print('Creating column 1...')
def column_01(df):
	df['ifgram_75_corrcoef_variance'] = df['filepath'].apply(lambda x: np.percentile(np.corrcoef(sps.describe(librosa.core.ifgram(librosa.core.load(x)[0])).variance),75))
	return df
df = parallelize_dataframe(df, column_01)
print("Creation of column took", time.time() - start_time, "seconds")
start_time = time.time()

print('Creating column 2...')
def column_02(df):
	df['ifgram_25_corrcoef_mean'] = df['filepath'].apply(lambda x: np.percentile(np.corrcoef(sps.describe(librosa.core.ifgram(librosa.core.load(x)[0])).mean),25))
	return df
df = parallelize_dataframe(df, column_02)
print("Creation of column took", time.time() - start_time, "seconds")
start_time = time.time()

print('Creating column 3...')
def column_03(df):
	df['ifgram_ptp_kurtosis'] = df['filepath'].apply(lambda x: np.ptp(sps.describe(librosa.core.ifgram(librosa.core.load(x)[0])).kurtosis))
	return df
df = parallelize_dataframe(df, column_03)
print("Creation of column took", time.time() - start_time, "seconds")
start_time = time.time()

print('Creating column 4...')
def column_04(df):
	df['ifgram_ptp_cov_kurtosis'] = df['filepath'].apply(lambda x: np.ptp(np.cov(sps.describe(librosa.core.ifgram(librosa.core.load(x)[0])).kurtosis)))
	return df
df = parallelize_dataframe(df, column_04)
print("Creation of column took", time.time() - start_time, "seconds")
start_time = time.time()

print('Creating column 5...')
def column_05(df):
	df['ifgram_25_cov_kurtosis'] = df['filepath'].apply(lambda x: np.percentile(np.cov(sps.describe(librosa.core.ifgram(librosa.core.load(x)[0])).kurtosis),25))
	return df
df = parallelize_dataframe(df, column_05)
print("Creation of column took", time.time() - start_time, "seconds")
start_time = time.time()

print('Creating column 6...')
def column_06(df):
	df['ifgram_std_cov_kurtosis'] = df['filepath'].apply(lambda x: np.std(np.cov(sps.describe(librosa.core.ifgram(librosa.core.load(x)[0])).kurtosis)))
	return df
df = parallelize_dataframe(df, column_06)
print("Creation of column took", time.time() - start_time, "seconds")
start_time = time.time()

print('Creating column 7...')
def column_07(df):
	df['chroma_cqt_mean_minmax'] = df['filepath'].apply(lambda x: np.mean(sps.describe(librosa.feature.chroma_cqt(librosa.core.load(x)[0])).minmax))
	return df
df = parallelize_dataframe(df, column_07)
print("Creation of column took", time.time() - start_time, "seconds")
start_time = time.time()

print('Creating column 8...')
def column_08(df):
	df['tonnetz_min_variance'] = df['filepath'].apply(lambda x: np.min(sps.describe(librosa.feature.tonnetz(librosa.core.load(x)[0])).variance))
	return df
df = parallelize_dataframe(df, column_08)
print("Creation of column took", time.time() - start_time, "seconds")
start_time = time.time()

print('Creating column 9...')
def column_09(df):
	df['tonnetz_ptp_variance'] = df['filepath'].apply(lambda x: np.ptp(sps.describe(librosa.feature.tonnetz(librosa.core.load(x)[0])).variance))
	return df
df = parallelize_dataframe(df, column_09)
print("Creation of column took", time.time() - start_time, "seconds")
start_time = time.time()

print('Creating column 10...')
def column_10(df):
	df['tempogram_ptp_variance'] = df['filepath'].apply(lambda x: np.ptp(sps.describe(librosa.feature.tempogram(librosa.core.load(x)[0])).variance))
	return df
df = parallelize_dataframe(df, column_10)
print("Creation of column took", time.time() - start_time, "seconds")
start_time = time.time()

print('Creating column 11...')
def column_11(df):
	df['tempogram_std_variance'] = df['filepath'].apply(lambda x: np.std(sps.describe(librosa.feature.tempogram(librosa.core.load(x)[0])).variance))
	return df
df = parallelize_dataframe(df, column_11)
print("Creation of column took", time.time() - start_time, "seconds")
start_time = time.time()

print('Creating column 12...')
def column_12(df):
	df['tempogram_var_variance'] = df['filepath'].apply(lambda x: np.var(sps.describe(librosa.feature.tempogram(librosa.core.load(x)[0])).variance))
	return df
df = parallelize_dataframe(df, column_12)
print("Creation of column took", time.time() - start_time, "seconds")
start_time = time.time()

print('Creating column 13...')
def column_13(df):
	df['tempogram_max_cov_variance'] = df['filepath'].apply(lambda x: np.max(np.cov(sps.describe(librosa.feature.tempogram(librosa.core.load(x)[0])).variance)))
	return df
df = parallelize_dataframe(df, column_13)
print("Creation of column took", time.time() - start_time, "seconds")
start_time = time.time()

print('Creating column 14...')
def column_14(df):
	df['tempogram_ptp_mean'] = df['filepath'].apply(lambda x: np.ptp(sps.describe(librosa.feature.tempogram(librosa.core.load(x)[0])).mean))
	return df
df = parallelize_dataframe(df, column_14)
print("Creation of column took", time.time() - start_time, "seconds")
start_time = time.time()

print('Creating column 15...')
def column_15(df):
	df['tempogram_std_mean'] = df['filepath'].apply(lambda x: np.std(sps.describe(librosa.feature.tempogram(librosa.core.load(x)[0])).mean))
	return df
df = parallelize_dataframe(df, column_15)
print("Creation of column took", time.time() - start_time, "seconds")
start_time = time.time()

print('Creating column 16...')
def column_16(df):
	df['tempogram_var_mean'] = df['filepath'].apply(lambda x: np.var(sps.describe(librosa.feature.tempogram(librosa.core.load(x)[0])).mean))
	return df
df = parallelize_dataframe(df, column_16)
print("Creation of column took", time.time() - start_time, "seconds")
start_time = time.time()

print('Creating column 17...')
def column_17(df):
	df['tempogram_max_cov_mean'] = df['filepath'].apply(lambda x: np.max(np.cov(sps.describe(librosa.feature.tempogram(librosa.core.load(x)[0])).mean)))
	return df
df = parallelize_dataframe(df, column_17)
print("Creation of column took", time.time() - start_time, "seconds")
start_time = time.time()

print('Creating column 18...')
def column_18(df):
	df['tempogram_ptp_skewness'] = df['filepath'].apply(lambda x: np.ptp(sps.describe(librosa.feature.tempogram(librosa.core.load(x)[0])).skewness))
	return df
df = parallelize_dataframe(df, column_18)
print("Creation of column took", time.time() - start_time, "seconds")
start_time = time.time()

print('Creating column 19...')
def column_19(df):
	df['tempogram_var_skewness'] = df['filepath'].apply(lambda x: np.var(sps.describe(librosa.feature.tempogram(librosa.core.load(x)[0])).skewness))
	return df
df = parallelize_dataframe(df, column_19)
print("Creation of column took", time.time() - start_time, "seconds")
start_time = time.time()

print('Creating column 20...')
def column_20(df):
	df['tempogram_var_kurtosis'] = df['filepath'].apply(lambda x: np.var(sps.describe(librosa.feature.tempogram(librosa.core.load(x)[0])).kurtosis))
	return df
df = parallelize_dataframe(df, column_20)
print("Creation of column took", time.time() - start_time, "seconds")
start_time = time.time()

print('Creating column 21...')
def column_21(df):
	df['tempogram_max_cov_kurtosis'] = df['filepath'].apply(lambda x: np.max(np.cov(sps.describe(librosa.feature.tempogram(librosa.core.load(x)[0])).kurtosis)))
	return df
df = parallelize_dataframe(df, column_21)
print("Creation of column took", time.time() - start_time, "seconds")
start_time = time.time()

print('Creating column 22...')
def column_22(df):
	df['ifgram_25_corrcoef_variance'] = df['filepath'].apply(lambda x: np.percentile(np.corrcoef(sps.describe(librosa.core.ifgram(librosa.core.load(x)[0])).variance),25))	
	return df
df = parallelize_dataframe(df, column_22)
print("Creation of column took", time.time() - start_time, "seconds")
start_time = time.time()

# Exporting dataframe to file
df.to_csv('df.csv')

# Operationalizing data
x = df.drop(['filepath'], axis=1)
columns = pd.DataFrame(x.columns, columns=['columns'])

# Converting complex values to float
for column in x.columns:
	if x[column].dtype == object:
		x[column] = x[column].apply(complex)

for column in x.columns:
	if x[column].dtype == complex:
		x[column] = x[column].apply(lambda x: x.real)

# Getting ranked predict_proba
predictions = pd.DataFrame(df['filepath'], columns=['filepath']).join(pd.DataFrame(model.predict_proba(x), columns=['0','1']))
proba_sort = predictions.sort_values('1', ascending=False)
ranked_files = proba_sort['filepath'].values

# Copying top 32 files to new folder
for x in range(32):
    shutil.copy2(ranked_files[x], 'assets/foreigner/selections/%s.wav' % '{0:03d}'.format(x + 1))