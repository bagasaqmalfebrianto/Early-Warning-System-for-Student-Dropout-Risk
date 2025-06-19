import joblib
import xgboost as xgb
from xgboost import XGBClassifier
 
model = joblib.load("model/xgb_model.joblib")
result_target = joblib.load("model/encoder_target.joblib")

def prediction(data):
    """Making prediction
 
    Args:
        data (Pandas DataFrame): Dataframe that contain all the preprocessed data
 
    Returns:
        str: Prediction result (Good, Standard, or Poor)
    """
    

    # model = XGBClassifier() 
    result = model.predict(data)
    final_result = result_target.inverse_transform(result)[0]
    return final_result