# mlops-assignment-1

## Problem statement & dataset
**Problem:** Multiclass classification on the Iris dataset.  
**Dataset:** Iris (150 samples, 4 features: sepal length/width, petal length/width; 3 classes).  
**Location:** Using scikit-learn `load_iris()` in `src/train_compare_mlflow.py`.

## Model Monitoring & Registration
- Used MLflow to log training runs with accuracy, precision, recall, and confusion matrix.
- Monitored runs in MLflow UI at `http://127.0.0.1:5000`.
- Selected the best model based on accuracy.
- Registered the best model in MLflow Model Registry with the name `BestModel_SVM`, Version 1.
