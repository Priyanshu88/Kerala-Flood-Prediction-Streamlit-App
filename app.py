import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Load data
@st.cache(allow_output_mutation=True)
def load_data():
    data = pd.read_csv('Kerala flood data.csv')
    return data

# Group the data by year and calculate the mean annual rainfall index for each year
data = load_data()

# show the data with describe

annual_rainfall_by_year = data.groupby("YEAR")[" ANNUAL RAINFALL"].mean()

# Create a bar chart to visualize the annual rainfall index for each year
plt.bar(annual_rainfall_by_year.index, annual_rainfall_by_year.values)
plt.xlabel("Year")
plt.ylabel("Annual Rainfall Index")
plt.title("Annual Rainfall Index by Year")
plt.show()


# Data preprocessing
data = data.drop(["SUBDIVISION"], axis=1) # Remove subdivision column
data = data.fillna(method="ffill") # Fill missing values with the last known value

# Split data into training and testing sets
X = data.drop(["FLOODS"], axis=1)
y = data["FLOODS"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train, y_train)
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)

# Define function for prediction
def predict(model, year, rainfall):
    # Create a dataframe with a single row containing the input values
    df = pd.DataFrame([[year] + rainfall], columns=X.columns)
    # Make prediction and return the result
    return model.predict(df)[0]

# Define Streamlit app
st.title("Kerala Flood Prediction")
st.write("This app predicts the likelihood of floods in Kerala based on rainfall data.")

# Show data
st.header("Data")
st.write(data)
st.header("Dataset Summary")
st.write(data.describe())

# Show model accuracy
st.header("Model Accuracy")
st.write("Decision Tree Classifier accuracy:", accuracy_score(y_test, dtc.predict(X_test)))
st.write("Random Forest Classifier accuracy:", accuracy_score(y_test, rfc.predict(X_test)))

# Get user input
st.header("Predict Floods")
year = st.number_input("Year", value=2021)
jan = st.number_input("January rainfall index", value=0.0)
feb = st.number_input("February rainfall index", value=0.0)
mar = st.number_input("March rainfall index", value=0.0)
apr = st.number_input("April rainfall index", value=0.0)
may = st.number_input("May rainfall index", value=0.0)
jun = st.number_input("June rainfall index", value=0.0)
jul = st.number_input("July rainfall index", value=0.0)
aug = st.number_input("August rainfall index", value=0.0)
sep = st.number_input("September rainfall index", value=0.0)
oct = st.number_input("October rainfall index", value=0.0)
nov = st.number_input("November rainfall index", value=0.0)
dec = st.number_input("December rainfall index", value=0.0)

# Make prediction
if st.button("Predict"):
    rainfall = [jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec, sum([jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec])]
    dtc_prediction = predict(dtc, year, rainfall)
    rfc_prediction = predict(rfc, year, rainfall)
    st.write(f"Decision Tree Prediction: {dtc_prediction}")
    st.write(f"Random Forest Prediction: {rfc_prediction}")



# Visualization and findings

# Load the dataset
df = pd.read_csv('Kerala flood data.csv')

df["FLOODS"]=df["FLOODS"].replace('YES',1)
df["FLOODS"]=df["FLOODS"].replace('NO',0)
df["FLOODS"]=df["FLOODS"].astype(int)
# Add a column to classify floods as 1 and non-floods as 0
df['is_flood'] = df['FLOODS'].apply(lambda x: 1 if x>0 else 0)

# Create visualisations
st.sidebar.title("Visualisations")

# Show a histogram of floods vs non-floods
fig1, ax1 = plt.subplots()
sns.histplot(data=df, x="is_flood", kde=True, ax=ax1)
st.sidebar.pyplot(fig1)

# Show a bar plot of annual rainfall vs floods
fig2, ax2 = plt.subplots()
df[" ANNUAL RAINFALL"] = df[" ANNUAL RAINFALL"].astype("float64")
sns.barplot(data=df, x="is_flood", y=" ANNUAL RAINFALL", ax=ax2)
st.sidebar.pyplot(fig2)

# Show a scatter plot of annual rainfall vs floods
fig3, ax3 = plt.subplots()
sns.scatterplot(data=df, x=" ANNUAL RAINFALL", y="FLOODS", hue="is_flood", ax=ax3)
st.sidebar.pyplot(fig3)

# Prepare the data for machine learning models
X = df.iloc[:, 2:-2]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a function to train and evaluate models
def train_evaluate_model(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    return acc, conf_matrix, class_report

# Create the models
decision_tree = DecisionTreeClassifier(random_state=42)
random_forest = RandomForestClassifier(random_state=42)

# Train and evaluate the models
dt_acc, dt_conf_matrix, dt_class_report = train_evaluate_model(decision_tree)
rf_acc, rf_conf_matrix, rf_class_report = train_evaluate_model(random_forest)

# Show the results
st.header("Model Results")
st.subheader("Decision Tree")
st.write(f"Accuracy: {dt_acc}")
st.write("Confusion Matrix:")
st.write(dt_conf_matrix)
st.write("Classification Report:")
st.write(dt_class_report)

st.subheader("Random Forest")
st.write(f"Accuracy: {rf_acc}")
st.write("Confusion Matrix:")
st.write(rf_conf_matrix)
st.write("Classification Report:")
st.write(rf_class_report)



