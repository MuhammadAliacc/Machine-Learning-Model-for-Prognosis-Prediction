# Machine-Learning-Model-for-Prognosis-Prediction
This project presents an in-depth analysis of prognosis data using the Random Forest Classifier, an advanced machine learning technique. The primary goal is to develop a robust predictive model to forecast patient outcomes based on various prognostic factors. The project leverages the power of Python, particularly the Scikit-learn library, and the spotpython library for hyperparameter optimization.

# Motivation
Accurate prognosis is crucial in healthcare to guide medical decision-making and improve patient outcomes. The motivation behind this project is to develop a predictive model that can aid healthcare professionals in making informed decisions for their patients. The Random Forest Classifier is known for its versatility, efficiency, and ability to handle complex data, making it an ideal choice for this task.

# Data set used 
The **Vector Borne Disease Dataset** was used in a study to predict medical prognosis. It consisted of hundreds of samples with case-specific features. The dataset included a target variable prognosis representing prognostic outcomes divided into **eleven classes**. To prepare the dataset for training a classifier model, preprocessing steps involved encoding prognosis names and performing feature engineering. **The goal was to predict the prognosis for unknown data based on the trained model.** 

# Data preprocessing (Feacher Engineering)
Before fitting the model, we conducted a series of essential data preprocessing steps to ensure the best performance of our Model. 
These steps include:
* Feature Combination: We combined relevant features to create more informative and higher-level features, enhancing the model's ability to capture complex relationships.
* Feature Clustering: Employing unsupervised learning techniques, we clustered similar features to reduce dimensionality and enhance the interpretability of the model.
* Feature Selection: Using various selection methods such as recursive feature elimination and statistical tests, we identified the most important features that significantly contribute to the prediction task.

# K-Fold Cross-Validation:
To robustly evaluate the performance of our model, we adopted the k-fold cross-validation method. The dataset was divided into k subsets (folds), and we trained and evaluated the model k times, each time using a different fold as the validation set and the remaining k-1 folds as the training set. This approach helps us to obtain more reliable performance metrics and minimize overfitting.

# Usage
To reproduce the results or apply the model to other prognosis datasets, clone the GitHub repository. You can customize the Random Forest Classifier's hyperparameters using spotpython or explore further variations of the preprocessing techniques.


#
### Libraries Used
* [spotpython](https://www.gm.th-koeln.de/~bartz/site/) : Used for Hperparameter Tuning
* [NumPy](https://numpy.org/) : Fundamental package for scientific computing
* [pandas](https://pandas.pydata.org/) : Used for manipulation and analysis of dataframes
* [scikit-learn](https://scikit-learn.org/stable/) : Library used to implement machine learning, and related methods
* [TensorFlow](https://www.tensorflow.org/) : Used for AI based models and methods

# 
### Contributors to the Source Code
* [Yuganshu Wadhwa](https://github.com/YuganshuWadhwa) 
* [Maximilian Brandt](https://github.com/brandeyy) 
* [Muhammad Ali](https://github.com/MuhammadAliacc) 
