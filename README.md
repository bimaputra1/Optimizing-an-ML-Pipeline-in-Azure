# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains data about direct phone call marketing of a Portuguese banking institution. 

In this project we aim to predict whether the customer decided to subrcribe to the bank's product or not.

The experiment compared both hyperdrive run and automl run. As a result, the best performing model was the AutoML method and Voting Ensemble model with an accuracy of 0.9173

## Scikit-learn Pipeline

**The pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

The pipeline was followed this order:
- Import Data
- Cleaning Data
- Data Split for training and testing
- Setting sampling parameter
- Create instance for logistic regression
- Model fitting
- Checking model accuracy
- Export best selected model

The parameters C and max_iter were used to tune the hyperparameter. C is an inverse of regularization strength, where smaller values specify stronger regularization, while max_iter specifies the number of maximum iterations that can be taken for the solvers to converge.

This experiment used Logistic regression as classification algorithm, since it is a good type of model for binary classification. It transforms its output using the logistic sigmoid function to return a probability value.

**The benefits of the random sampling method**

The random sampling method was chosen in this case because of the limited time available to do the run. This method uses less computation resources and takes less time, compared to other sampling methods such as grid sampling. This is due to the fact that random sampling randomly selects hyperparameter values from the defined seach space, while grid sampling performs a simple but exhaustive grid search over all possible values.

**The benefits of the Bandit policy as early stopping policy**

The Bandit policy saves time and computation resources in running models that do not perform well, since it stops those models early, as soon as they are showing bad results.

## AutoML
AutoML ran various algorithms with relatively simple code.In totals, the AutoML ran and generates 25 models. These model then compared based on the results to decided the best model. In this case, the choosen best model was "Voting Ensemble".

Voting ensemble is one example of ensemble learning methods that combines different algorithms results and the results was the most voted class from the different models.

## Pipeline comparison

The experiment was used both hyperdrive run and automl run. As a result, the best performing model for each methoeds were:
- Logistic Regression with a C value of 2.0 and max_iter of 50 which get  accuracy 0.912
- Best algorithm with AutoML was Voting Ensemble with an accuracy of 0.9173

So, we can decide that the best model is with an accuracy of

Both method give us fast and efficient way to find a best algorithm and tuning hyperparameter. However, AutoML performs better since it choose different types of models to find the best algorithm. On the other hand, the Hyperdrive only fine tuned the hyperparameter with one algorithm.

To get better results, these two pipeline can be combined. First, we find the best algorithm. Then we perform hyperdrive to find the best hyperparameter from this model.

## Future work
Some improvement that can be considered for the next experiment:
- Fixing imbalance dataset prior to building the models. This can be done by performing SMOTE technique or under/over sampling method.
- Performs the Hyperdrive run to fine tuned the best model generated in AutoML.
- Changes the random parameter sampling values.


## Proof of cluster clean up
The cluster clean up code was included in the notebook and excecuted at the end of the code.
