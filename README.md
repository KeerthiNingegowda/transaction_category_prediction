# transaction_category_prediction
Predict the category of given a transaction description

### To activate the virtual environment - Assuming uv is already installed
```
uv sync 
source .venv/bin/activate
```

Note:- I ran n_trials=5 in train function. Change it if you want

### To run the project 
```python main.py```

### For research, EDA, design choices, assumptioons and future enhancements checkout 
[Design choices](https://github.com/KeerthiNingegowda/transaction_category_prediction/blob/main/design-choices.pdf)

### Feature importance
model is trained only on one feature. So not applicable 

### Deployment in real world consideration (summary) This is wildly problem dependent
1) Data pipeline hardening:- 
- Having agreement on frequency of data pulling from databases and the format of storage. For eg:- parquet is more efficient than csv especially fro production
- Data versioning, if applicable
2) Model training pipeline hardening:- Establish the following
- Model versioning - Required for tracking changes in the model both for team and compliance purposes
- Model repository - like mlflow or weights/biases
- Retraining pipeline - can  be done using airflow 
- Model performance monitoring - Can be done using airflow or dashboard like tableau or kibana, having a holdout set or asking the user for human evaluation once in a while
- Configuration management using packages like hydra - if applicable
- Proper Github permissions for model training code along with a good review process
3) Deployment considerations - MLOps
- Models and model code can be packaged into Docker
- Consider the management of end to end environment handling, pipeline handling etc., usinh github actions
- Write applicable tests  i.e. User Acceptance testing, smoke testing, regression testing when trying to deploy model to production
- Create API endpoints depending on what service you are offering probably using Flask
- Implement Post Implementation Verification , if applicable
