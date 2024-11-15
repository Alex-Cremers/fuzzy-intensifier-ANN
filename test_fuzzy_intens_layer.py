# This is a simple test of the architecture on a dataset from kaggle

import pandas as pd

# Setup plotting
import matplotlib.pyplot as plt
# plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer


import tensorflow as tf
import keras
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from keras import layers

# Import custom op and layer definition:
# intens_weighting_module = tf.load_op_library("./intens_weighting.so")
# intens_weighting = intens_weighting_module.intens_weighting
# from register_intens_weighting_grad import *
from fuzzy_intens_layer import *


tf.config.threading.set_inter_op_parallelism_threads(24)
tf.config.threading.set_intra_op_parallelism_threads(24)

# Read data
hotel = pd.read_csv('./hotel.csv')


# Preprocess (just copied from kaggle, but with MinMaxScaler)
X = hotel.copy()
y = X.pop('is_canceled').astype(float)

X['arrival_date_month'] = \
    X['arrival_date_month'].map(
        {'January':1, 'February': 2, 'March':3,
         'April':4, 'May':5, 'June':6, 'July':7,
         'August':8, 'September':9, 'October':10,
         'November':11, 'December':12}
    )

features_num = [
    "lead_time", "arrival_date_week_number",
    "arrival_date_day_of_month", "stays_in_weekend_nights",
    "stays_in_week_nights", "adults", "children", "babies",
    "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "required_car_parking_spaces",
    "total_of_special_requests", "adr",
]
features_cat = [
    "hotel", "arrival_date_month", "meal",
    "market_segment", "distribution_channel",
    "reserved_room_type", "deposit_type", "customer_type",
]

transformer_num = make_pipeline(
    SimpleImputer(strategy="constant"), # there are a few missing values
    MinMaxScaler(),
)
transformer_cat = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="NA"),
    OneHotEncoder(handle_unknown='ignore'),
)

preprocessor = make_column_transformer(
    (transformer_num, features_num),
    (transformer_cat, features_cat),
)

# stratify - make sure classes are evenlly represented across splits
X_train, X_valid, y_train, y_valid = \
    train_test_split(X, y, stratify=y, train_size=0.75)

X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)

# Check for NaN in data:
# np.any(pd.DataFrame(X_train).isna())
# np.any(pd.DataFrame(X_valid).isna())
# np.any(y_train.isna())
# np.any(y_valid.isna())

# Define model
model = keras.Sequential([
    FuzzyIntens(12, kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3), name = 'first_fuzzy_layer'),
    FuzzyIntens(4, kernel_regularizer=tf.keras.regularizers.L1(l1=1e-3), name = 'hidden_fuzzy_layer'),
    FuzzyIntens(1, kernel_regularizer=tf.keras.regularizers.L1(l1=1e-3), name = 'outer_fuzzy_layer')
])

# Check that we can do a forward pass:
y_init = model(X_train)


# Check properties of the output:
y_init.numpy().min()
y_init.numpy().mean()
y_init.numpy().max()



# Inspect intermediate layer:
model.summary()
model_part = keras.Model(
  inputs=model.get_layer("first_fuzzy_layer").input,
  outputs=model.get_layer("first_fuzzy_layer").output)


model.compile(
    optimizer=tf.keras.optimizers.Adam(), # clipnorm=50., 
    loss='mean_squared_error', # 'cosine_similarity',
    jit_compile=False # Avoid XLA until we can figure out what goes wrong
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=50,
    verbose=1, 
)

model.summary()

model.save("hotel_fit.keras")

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Cosine similarity")

#.savefig("fit1_accuracy.pdf", format = "pdf")

history_df


y_trained = model(X_train)
y_trained.numpy()[:,0]

pd.DataFrame(model.get_weights()[0]).describe()
np.any([np.any(np.isnan(x)) for x in model.get_weights()])

# Is any weight NaN?
y_trained_interm = model_part(X_train)

y_trained.numpy().max()
df = pd.DataFrame({'fitted':y_trained[:,0], 'actual':y_train})
df.query('actual==1.0').value_counts().iloc[1:15]
df.query('actual==0.0').value_counts().iloc[1:15]
df.groupby('actual')['fitted'].mean()
df.query('fitted == 0.0|fitted == 1.0').groupby('actual')['fitted'].mean()
df.query('fitted == 0.0|fitted == 1.0').value_counts()

