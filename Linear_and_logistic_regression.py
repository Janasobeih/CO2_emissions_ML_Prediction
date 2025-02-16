import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


# Loading the dataset
def loadingDataSet(path):
    dataset = pd.read_csv(path)
    return dataset


# Checking if there is any missing values
def checkingForMissingValues(dataset):
    missingValues = dataset.isnull().sum()
    # print("The Number Of Missing values in Each Column:")
    # print(missingValues)
    return missingValues


# Function to check if numeric features have the same scale
def normalizingData(dataset):
    # print("The Descriptive data for each numeric column:")
    # print(dataset.describe())
    minMaxScaler = MinMaxScaler()

    # Selecting the columns with numeric values
    columnsWithNumericValues = dataset.select_dtypes(include=['number']).columns

    # Normalizing the columns with numeric values
    dataset[columnsWithNumericValues] = minMaxScaler.fit_transform(dataset[columnsWithNumericValues])
    # Displaying the data set after being normalized
    # print("Numeric Columns after Normalization:")
    # print(dataset.select_dtypes(include=['number']))

    return dataset


# Function to visualize the pairplot
def pairPlotting(dataset):
    sns.pairplot(dataset)
    plt.title("pairplot")
    plt.show()


# Function to visualize the correlation heatmap
def correlationHeatmapping(dataset):
    numeric_columns = dataset.select_dtypes(include=['number'])

    correlation_matrix = numeric_columns.corr()

    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Heatmap")
    plt.show()


def separatingDataSet(dataset):
    features = dataset[['Make', 'Model', 'Vehicle Class', 'Engine Size(L)', 'Cylinders', 'Transmission', 'Fuel Type',
                        'Fuel Consumption City (L/100 km)', 'Fuel Consumption Hwy (L/100 km)',
                        'Fuel Consumption Comb (L/100 km)', 'Fuel Consumption Comb (mpg)']]
    targets = dataset[['CO2 Emissions(g/km)', 'Emission Class']]
    # print("features columns:")
    # print(features)
    # print("Target columns:")
    # print(targets)
    return features, targets


# this function encodes the categorical features and targets
def encodingFeatures(dataset):
    categoricalFeatures = dataset[['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type', ]]
    labelEncoder = LabelEncoder()
    for feature in categoricalFeatures:
        dataset[feature] = labelEncoder.fit_transform(dataset[feature])

    target = 'Emission Class'
    dataset[target] = labelEncoder.fit_transform(dataset[target])
    return categoricalFeatures, target


def splittingandScaling(dataset):
    x, y = separatingDataSet(dataset)
    featuresTrain, featuresTest, targetTrain, targetTest = train_test_split(x, y, test_size=0.25, random_state=123)

    # Applying scaling to the training and testing features
    X_train_scaled, X_test_scaled = scalingAfterSplitting(featuresTrain, featuresTest)

    print(f"Training Features(X) Shape: {X_train_scaled.shape}")
    print(f"Testing Features(X) Shape: {X_test_scaled.shape}")
    print(f"Training Target(Y) Shape: {targetTrain.shape}")
    print(f"Testing Target(Y) Shape: {targetTest.shape}")

    return featuresTrain, featuresTest, targetTrain, targetTest


# scaling numeric features after splitting the data
def scalingAfterSplitting(X_train, X_test):
    # Separate numeric columns for scaling
    numeric_columns = X_train.select_dtypes(include=['number']).columns

    # Initialize the MinMaxScaler
    minMaxScaler = MinMaxScaler()

    # Only scale the numeric columns
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_columns] = minMaxScaler.fit_transform(X_train[numeric_columns])
    X_test_scaled[numeric_columns] = minMaxScaler.transform(X_test[numeric_columns])

    return X_train_scaled, X_test_scaled


################################################################################################## shoghl salma


# Point d
# Function to find correlation between given features and
# the Co2 emission
# Also find the correlation between each and every feature
# then choose two features

def findCorrelation(dataset):
    numericFeatures = dataset.select_dtypes(include=[np.number])

    # correlation matrix for numeric features for each feature with the other and printing it
    correlation_matrix = numericFeatures.corr()

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    print("Pairwise Feature Correlations:")
    print(correlation_matrix)

    # correlation matrix for each feature with the target feature (CO2 Emissions) and printing it
    target = 'CO2 Emissions(g/km)'
    correlations = numericFeatures.corr()[target].sort_values(ascending=False)
    print("-------------------------------------------------")
    print("-------------------------------------------------")

    print("Feature Correlations with CO2 Emissions:")
    print(correlations)

    bestPair = ["Fuel Consumption City (L/100 km)", "Engine Size(L)"]
    print("Best Pair:", bestPair)

    # Create a heatmap for the full correlation matrix (including the target)
    correlationHeatmapping(dataset)

    # Create a heatmap for correlations with the target
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlations.drop(target).to_frame(), annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
    plt.title("Correlation Heatmap of Features with CO2 Emissions (Excluding CO2 Emissions)")
    plt.show()

    return bestPair

#calculates the gradient descent for each theta for the given number of iterations with the given learning rate
#also calculates cost function (percenatge of the error) for each iteration
#plots the cost of each iteration on the graph
#that later shows the decrease in cost
def costCalculation(dataset, bestPair, target, iterations=1500, learningRate=0.01):
    dataset = normalizingData(dataset)

    feature1 = dataset[bestPair[0]].values
    feature2 = dataset[bestPair[1]].values
    y_actual = dataset[target].values
    m = len(y_actual)

    theta_0 = 0
    theta_1 = 0
    theta_2 = 0

    # array to store all the costs
    costs = []

    # Gradient descent loop
    for iter in range(iterations):
        y_predicted = theta_0 + theta_1 * feature1 + theta_2 * feature2
        #gradient descent for each theta separately
        gradient_theta_0 = -(1 / m) * np.sum(y_actual - y_predicted)
        gradient_theta_1 = -(1 / m) * np.sum((y_actual - y_predicted) * feature1)
        gradient_theta_2 = -(1 / m) * np.sum((y_actual - y_predicted) * feature2)

        theta_0 -= learningRate * gradient_theta_0
        theta_1 -= learningRate * gradient_theta_1
        theta_2 -= learningRate * gradient_theta_2

        cost = (1 / (2 * m)) * np.sum((y_actual - y_predicted) ** 2)
        costs.append(cost)
        #prints the cost of every hundred iterations
        if iter % 100 == 0:
            print(f"Iteration {iter}: Cost = {cost}")
    #plot that shows the learning rate over the iterations
    plt.figure(figsize=(10, 6))
    plt.plot(range(iterations), costs, label="Cost")
    plt.xlabel("Iteration")
    plt.ylabel("Cost (MSE)")
    plt.title("Cost Reduction Over Iterations of Gradient Descent (Multivariate)")
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f"Final Theta_0 (Intercept): {theta_0}")
    print(f"Final Theta_1 (Coefficient for {bestPair[0]}): {theta_1}")
    print(f"Final Theta_2 (Coefficient for {bestPair[1]}): {theta_2}")

    return theta_0, theta_1, theta_2, costs


def evaluateModelWithR2(dataset, bestPair, target, theta_0, theta_1, theta_2):

    feature1 = dataset[bestPair[0]].values
    feature2 = dataset[bestPair[1]].values
    y_actual = dataset[target].values

    y_predicted = theta_0 + theta_1 * feature1 + theta_2 * feature2

    r2 = r2_score(y_actual, y_predicted)
    print(f"R^2 Score on Test Set: {r2:.4f}")


def linearRegression(dataset):

    target = 'CO2 Emissions(g/km)'

    bestPair = findCorrelation(dataset)

    train_data, test_data = train_test_split(dataset, test_size=0.25, random_state=123)

    train_data = normalizingData(train_data)
    test_data = normalizingData(test_data)

    theta_0, theta_1, theta_2, costs = costCalculation(train_data, bestPair, target)

    evaluateModelWithR2(test_data, bestPair, target, theta_0, theta_1, theta_2)

    ################################################################################################

    #point e

# sigmoid function
def sigmoid(i):
    i = np.clip(i, -500, 500)
    return 1 / (1 + np.exp(-i))

#the prediction function to predict the values
def predict(x, theta, b):
    i = np.dot(x, theta) + b
    A = sigmoid(i)
    return np.where(A >= 0.5, 1, 0).reshape(-1)

#logistic regression function that uses the model returned from sgdc and prints the results
def LogisticRegressionResults(model, x_test, y_test):
    y_pred = model.predict(x_test)
    print("Predicted Labels:", y_pred)
    print("Actual Labels:", y_test.flatten())
    accuracy = (y_pred == y_test.flatten()).mean()
    print(f"Model Accuracy (fraction): {accuracy:.4f}")


# Main function to run all tasks
def main():
    # Loading the dataset
    dataset = loadingDataSet("C:/Users/Eng-Wael/PycharmProjects/MLAssig/co2_emissions_data.csv")

    """# Checking for missing values
    checkingForMissingValues(dataset)

    #scaling the features
    normalizingData(dataset)

    #Visualizing the data using pairplot
    pairPlotting(dataset)

    #Visualizing the correlation heatmap
    correlationHeatmapping(dataset)

    separatingDataSet(dataset)

    encodingFeatures(dataset)

    splittingandScaling(dataset)"""

    checkingForMissingValues(dataset)
    linearRegression(dataset)

    print("------Point e------------")

    dataset = normalizingData(dataset)
    encodingFeatures(dataset)
    x, y = separatingDataSet(dataset)
    y = y[['Emission Class']].values
    x = x.values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=123)

    model = SGDClassifier(loss='log_loss', max_iter=1000, learning_rate='constant', eta0=0.01)
    model.fit(x_train, y_train.flatten())

    print("Testing the Logistic Regression Model...")
    LogisticRegressionResults(model, x_test, y_test)


# Run the program
if __name__ == "__main__":
    main()