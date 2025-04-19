#importing essential libraries for data manipulation, visualization, and machine learning
import pandas as pd #library for data manipulation and analysis, handling datasets in DataFrame.
from sklearn.preprocessing import MinMaxScaler #Part of sklearn.preprocessing, this scaler transforms features to a fixed range, usually [0, 1], which is useful for normalization.
import matplotlib.pyplot as plt #library for creating static, interactive, and animated visualizations
import seaborn as sns #provides a high-level interface for drawing attractive and informative statistical graphics.
from imblearn.over_sampling import RandomOverSampler #handle class imbalance by oversampling the minority class.
from collections import Counter #counts the frequency of elements, useful for analyzing class distributions.
from sklearn.model_selection import train_test_split #Splits the dataset into training and testing subsets.
from sklearn.linear_model import LogisticRegression #Machine learning models from sklearn for classification tasks.
from sklearn.ensemble import RandomForestClassifier #Machine learning models from sklearn for classification tasks.
from sklearn.svm import SVC #Machine learning models from sklearn for classification tasks.
from sklearn.metrics import classification_report #Generates a detailed report of classification metrics such as precision, recall, and F1-score.
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout, MaxPooling1D
from tensorflow.keras.layers import InputLayer, LSTM, Dropout, Dense
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam



# Load the datasets
fake_accounts_df = pd.read_csv("fakeAccountData.csv")
real_accounts_df = pd.read_csv("realAccountData.csv")

# Merge the two datasets
merged_df = pd.concat([fake_accounts_df, real_accounts_df], ignore_index=True)

# Save the merged dataset to a new CSV file
merged_df.to_csv("mergedAccountData.csv", index=False)

# Remove missing values
cleaned_df = merged_df.dropna()

# Remove duplicate rows
cleaned_df = cleaned_df.drop_duplicates()

# Selecting numerical features to normalize
numerical_features = ['userFollowerCount', 'userFollowingCount', 'userBiographyLength', 'userMediaCount']

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Apply normalization
cleaned_df[numerical_features] = scaler.fit_transform(cleaned_df[numerical_features])

# Handle class imbalance using oversampling
X = cleaned_df.drop(columns=['isFake'])
y = cleaned_df['isFake']

oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)

balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
balanced_df['isFake'] = y_resampled

# Feature selection: Keep only top correlated features
selected_features = ['userHasProfilPic', 'userFollowingCount', 'usernameDigitCount', 'userIsPrivate', 'userFollowerCount', 'isFake']
balanced_df = balanced_df[selected_features]

# Save the selected features dataset
balanced_df.to_csv("selectedFeaturesAccountData.csv", index=False)

balanced_df.head()

# Visualizations
# A. Class Distribution (Bar Chart)
plt.figure(figsize=(6, 4))
class_counts = balanced_df['isFake'].value_counts()
fig, ax = plt.subplots(figsize=(6, 4))
bar_chart = ax.bar(class_counts.index, class_counts.values, tick_label=['Real (0)', 'Fake (1)'])

# Annotate bars with exact counts
for bar in bar_chart:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
            f'{int(bar.get_height())}', ha='center', va='center', fontsize=10, color='black')

plt.xlabel("Account Type")
plt.ylabel("Count")
plt.title("Class Distribution of Real and Fake Accounts After Balancing")
plt.show()

# B. Profile Picture Effect (Stacked Bar Chart)
profile_pic_counts = balanced_df.groupby(['isFake', 'userHasProfilPic']).size().unstack()
fig, ax = plt.subplots(figsize=(8, 5))
profile_pic_counts.plot(kind='bar', stacked=True, ax=ax)

# Annotate exact counts
for bars in ax.containers:
    ax.bar_label(bars, fmt='%d', label_type='center', fontsize=10, color='black')

plt.xlabel("Account Type (0 = Real, 1 = Fake)")
plt.ylabel("Count")
plt.title("Profile Picture Effect on Fake vs. Real Accounts")
plt.legend(title="Has Profile Picture", labels=["No (0)", "Yes (1)"])
plt.show()

# C. Username Digit Count Histogram
fig, ax = plt.subplots(figsize=(8, 5))
n, bins, patches = ax.hist([balanced_df[balanced_df['isFake'] == 0]['usernameDigitCount'],
                            balanced_df[balanced_df['isFake'] == 1]['usernameDigitCount']],
                           bins=range(0, balanced_df['usernameDigitCount'].max() + 2),
                           label=["Real Accounts (0)", "Fake Accounts (1)"], alpha=0.7, edgecolor="black")

# Annotate bars with exact counts
for patch_set in patches:
    for patch in patch_set:
        height = patch.get_height()
        if height > 0:
            ax.text(patch.get_x() + patch.get_width() / 2, height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10, color='black')

plt.xlabel("Number of Digits in Username")
plt.ylabel("Count")
plt.title("Username Digit Count Distribution for Real vs. Fake Accounts")
plt.legend()
plt.show()


# The data must be shuffled before applying ML and splitting to prevent bias
# Set a random seed for reproducibility
random_seed = 42

# Shuffle the columns using random seed
balanced_df = balanced_df.sample(frac=1, axis=1, random_state=random_seed)
balanced_df.head()

# Split the dataset into training (70%), validation (15%), and testing (15%) using stratified splitting
X_train, X_temp, y_train, y_temp = train_test_split(balanced_df.drop(columns=['isFake']), balanced_df['isFake'], test_size=0.3, stratify=balanced_df['isFake'], random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Save the split datasets
X_train.to_csv("X_train.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
X_val.to_csv("X_val.csv", index=False)
y_val.to_csv("y_val.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

# Develop machine learning models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

# Train and evaluate models
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Evaluation for {model_name}:")
    print(classification_report(y_test, y_pred, digits=4))
    print("-" * 50)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)



# Load the training and testing datasets
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").values.ravel()

# Define and train the MLP model
mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)

print("Training MLP Model...")
mlp_model.fit(X_train, y_train)

# Make predictions
y_pred = mlp_model.predict(X_test) #Uses the trained model to predict labels for the test dataset.

# Evaluate the model
print("MLP Model Evaluation:")
print(classification_report(y_test, y_pred, digits=4))


# Reshape data for CNN (adding a channel dimension)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Define CNN model
cnn_model = Sequential([
    Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(X_train.shape[1], 1)), #Extracts patterns from account-related numerical features, such as followers count, profile picture presence, and username digit count.
    MaxPooling1D(pool_size=2), #Keeps only the most relevant features, helping the model generalize better.
    Conv1D(filters=32, kernel_size=1, activation='relu'), #A second Conv1D layer applies 32 more filters to extract deeper relationships.
    MaxPooling1D(pool_size=2), #reduces dimensions to keep only essential patterns.
    Flatten(), #Prepares the extracted features for classification.
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
]) #Predicts whether an account is real or fake based on extracted features.

# Compile the model
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the CNN model
print("Training CNN Model...")
cnn_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# Make predictions
y_pred = (cnn_model.predict(X_test) > 0.5).astype("int32").ravel()

# Evaluate the model
print("CNN Model Evaluation:")
print(classification_report(y_test, y_pred, digits=4))


# Preprocessing steps (example)
scaler = MinMaxScaler()

# Ensure X_train and X_test are 2D before scaling
if len(X_train.shape) > 2:
    X_train = X_train.reshape((X_train.shape[0], -1))
if len(X_test.shape) > 2:
    X_test = X_test.reshape((X_test.shape[0], -1))

# Now apply MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 2: Reshape to 3D for LSTM
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Set timesteps and features
timesteps = 1
features = X_train_scaled.shape[1]

# Now build your LSTM model

lstm_model = Sequential()
lstm_model.add(InputLayer(input_shape=(timesteps, features)))
lstm_model.add(LSTM(64, activation='tanh', return_sequences=False))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(32, activation='relu'))
lstm_model.add(Dense(1, activation='sigmoid'))

lstm_model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the model
lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, verbose=1, validation_data=(X_test_lstm, y_test))

# Make predictions
y_pred = (lstm_model.predict(X_test_lstm) > 0.5).astype("int32").ravel()

# Evaluate the model
print("LSTM Model Evaluation:")
print(classification_report(y_test, y_pred, digits=4))




#Create interface
# ======= Section 1: Imports =======
import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report

#Saving the model:
joblib.dump(models["Random Forest"], 'random_forest_model.pkl')


# Load model
model = joblib.load("random_forest_model.pkl")

# ======= Section 2: Load and Preprocess Data =======
# Load your cleaned and selected feature dataset
df = pd.read_csv("selectedFeaturesAccountData.csv")

# Split features and labels
X = df.drop(columns=['isFake'])
y = df['isFake']

# Split data: 70% train, 15% val, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Fix input shape for classical models
if len(X_train.shape) == 3:
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]))

# ======= Section 3: Train and Save Random Forest =======
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

# Train and evaluate
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"== {model_name} ==")
    print(classification_report(y_test, y_pred, digits=4))
    print("-" * 50)

# Save the trained Random Forest model
joblib.dump(models["Random Forest"], "random_forest_model.pkl")

# ======= Section 4: Streamlit App with Navigation =======
# Load the saved Random Forest model
model = joblib.load("random_forest_model.pkl")

# Features needed by the model
selected_features = ['userHasProfilPic', 'userFollowingCount', 'usernameDigitCount', 'userIsPrivate', 'userFollowerCount']

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🧪 Fake Account Detection", "🤖 Bot Account Detection"])

# --- Preprocessing Function ---
def preprocess(df):
    scaler = MinMaxScaler()
    df[['userFollowingCount', 'userFollowerCount']] = scaler.fit_transform(df[['userFollowingCount', 'userFollowerCount']])
    return df

# --- Page 1: Home ---
if page == "🏠 Home":
    st.title("📊 Social Media Integrity Toolkit")
    st.markdown("""
        Welcome to the **Fake and Bot Account Detection Tool**!  
        Use this app to:
        - 📌 Detect suspicious or fake accounts.
        - 🤖 (Soon) Identify bot-like behavior on social media.

        Select a page from the sidebar to get started.
    """)

# --- Page 2: Fake Account Detection ---
elif page == "🧪 Fake Account Detection":
    st.title("🧪 Fake Account Detection")

    uploaded_file = st.file_uploader("📁 Upload CSV File for Batch Detection", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df = preprocess(df)
        df = df[selected_features]
        predictions = model.predict(df)
        df['Prediction'] = ['Fake' if p == 1 else 'Real' for p in predictions]
        st.success("✅ Prediction complete!")
        st.dataframe(df)
    else:
        st.subheader("🔍 Manual Input")
        userHasProfilPic = st.selectbox("Has Profile Picture?", [0, 1])
        userFollowingCount = st.number_input("Following Count", min_value=0)
        usernameDigitCount = st.number_input("Username Digit Count", min_value=0)
        userIsPrivate = st.selectbox("Is Private Account?", [0, 1])
        userFollowerCount = st.number_input("Follower Count", min_value=0)

        if st.button("Predict"):
            input_df = pd.DataFrame([{
                "userHasProfilPic": userHasProfilPic,
                "userFollowingCount": userFollowingCount,
                "usernameDigitCount": usernameDigitCount,
                "userIsPrivate": userIsPrivate,
                "userFollowerCount": userFollowerCount
            }])
            input_df = preprocess(input_df)
            prediction = model.predict(input_df)[0]
            st.success(f"🎯 Prediction: **{'Fake' if prediction == 1 else 'Real'}**")

# --- Page 3: Bot Account Detection ---
elif page == "🤖 Bot Account Detection":
    st.title("🤖 Bot Account Detection (Coming Soon)")

    st.markdown("""
        This feature will help identify accounts with automated or bot-like behavior.

        ✅ Status: In Progress  
        🛠️ If you have a bot dataset or model ready, you can upload it and extend this page.
    """)