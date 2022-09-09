# Model Training and Register

## Model Register
I'm using a sklearn library, mlflow provides a way to register the model with the following command:

'''
  #Model Register
  mlflow.sklearn.log_model(
        sk_model = logreg,
        artifact_path='models/logreg',
        registered_model_name='sk-learn-logreg-model'
  )
 '''
