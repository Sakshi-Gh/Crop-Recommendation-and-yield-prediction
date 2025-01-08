# Importing Libraries
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("data/crop_yield.csv")

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['crop_encoded'] = label_encoder.fit_transform(df['Crop'])
df['season_encoded'] = label_encoder.fit_transform(df['Season'])
df['state_encoded'] = label_encoder.fit_transform(df['State'])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df['Crop_Year_scaled'] = scaler.fit_transform(df[['Crop_Year']])
df['Area_scaled'] = scaler.fit_transform(df[['Area']])
df['Production_scaled'] = scaler.fit_transform(df[['Production']])
df['Annual_Rainfall_scaled'] = scaler.fit_transform(df[['Annual_Rainfall']])
df['Fertilizer_scaled'] = scaler.fit_transform(df[['Fertilizer']])
df['Pesticide_scaled'] = scaler.fit_transform(df[['Pesticide']])
df['Yield_scaled'] = scaler.fit_transform(df[['Yield']])

features = ['crop_encoded', 'state_encoded','Area_scaled','Annual_Rainfall_scaled','Fertilizer_scaled']
target   = ['Yield_scaled' ]

X = df.loc[: , features]
Y = df.loc[: , target]

from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size = 0.4,random_state = 20)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

X_train_ls = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_ls = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
Y_train_ls  = Y_train.values

print(X_train_ls.shape,X_train_ls[0])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

model_lstm = Sequential([
    Bidirectional(LSTM(units=200, return_sequences=True, input_shape=(X_train_ls.shape[1], X_train_ls.shape[2]))),
    Dropout(0.2),
    LSTM(units=100),
    Dropout(0.2),
    Dense(units=1)
])

optimizer = Adam(learning_rate=0.001)
model_lstm.compile(optimizer=optimizer, loss='mean_squared_error')
model_lstm.summary()

early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

history = model_lstm.fit(X_train_ls, Y_train_ls, epochs=200, batch_size=32, validation_split=0.2,
                    callbacks=[early_stopping], verbose=1)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
Y_pred = model_lstm.predict(X_test_ls)

mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)

print("Mean Absolute Error:", mae)
print("R-squared (R2):", r2)
print("Mean Squared Error:", mse)

# Save the LSTM model
model_lstm.save("lstm_model.h5")  # Saved as .h5 format, which is the standard format for Keras models

# Load the LSTM model
from tensorflow.keras.models import load_model
loaded_lstm_model = load_model("lstm_model.h5")

from sklearn.metrics import r2_score
global model_accuracy
Y_pred_test = loaded_lstm_model.predict(X_test_ls).flatten()
model_accuracy = r2_score(Y_test, Y_pred_test)
print(model_accuracy)