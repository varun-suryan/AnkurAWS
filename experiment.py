import requests
import base64
import pandas as pd
import uvicorn
from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from fastapi.responses import FileResponse

app = FastAPI()
class request_body_train(BaseModel):
    target: str

class pipeline:
    def __init__(self, url, token):
        self.url = url
        self.token = token
        self.router = APIRouter()
        self.router.add_api_route('/', self.home, methods=['GET'])
        self.router.add_api_route('/train', self.train, methods=['POST'])
        self.router.add_api_route('/results', self.results, methods=['GET'])

    def convert_to_dataframe(self, response):

        """Converts the response from the API to a pandas dataframe"""
        # get the binary text
        decodedBytes = base64.b64decode(response.text.split(',')[1]).split(b'\r\n')

        columns = [column.split('_')[-1] for column in decodedBytes[0].decode('ascii').split(',')]
        list_of_rows = []

        for row in decodedBytes[1:]:
            decoded_row = row.decode('ascii').split(',')
            list_row = []
            for item in decoded_row:
                try:
                    item = float(item)
                except:
                    pass
                list_row.append(item)
            if len(list_row) == len(columns):
                list_of_rows.append(list_row)

        return pd.DataFrame(list_of_rows, columns=columns)

    def home(self):
        return {"Hello! I am a model that predicts churn. To train me, send a POST request to /train with a target variable. To get my results, send a GET request to /results."}
    def results(self):
        return FileResponse('feature_importance.png')
        # cv2image = cv2.imread('feature_importance.png')
        # res, im_png = cv2.imencode(".png", cv2image)
        # print(StreamingResponse(im_png, media_type="image/png"))


    def feature_importance(self):
        categorical_columns = self.X_train.select_dtypes(include=['object', 'category']).columns

        categorical_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

        preprocessing = ColumnTransformer([
                ("cat", categorical_encoder, categorical_columns),])

        rf = Pipeline([("preprocess", preprocessing),("classifier", RandomForestClassifier(random_state=42)),])
        rf.fit(self.X_train, self.y_train)
        result = permutation_importance(rf, self.X_train, self.y_train, n_repeats=5, random_state=42, n_jobs=2)
        sorted_importances_idx = result.importances_mean.argsort()

        importances = pd.DataFrame(
            result.importances[sorted_importances_idx].T,
            columns= self.X_train.columns[sorted_importances_idx],
        )
        ax = importances.plot.box(vert=False, whis=10)
        ax.set_title("Churn Prediction")
        ax.axvline(x=0, color="k", linestyle="--")
        ax.set_xlabel("Importance in Churn Prediction")
        ax.figure.tight_layout()
        ax.grid();
        ax.figure.savefig('feature_importance.png')
        return None


    def train(self, data: request_body_train):
        # This is the URL and Bearer token of the API endpoint
        payload = {}
        headers = {"Authorization": self.token}
        df = self.convert_to_dataframe(requests.request("GET", self.url, headers=headers, data=payload))

        X = df.drop(['customerid', data.target, 'id'], axis=1)
        y = df[data.target].replace({'Yes': 1, 'No': 0})

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.feature_importance()

        self.X_train, self.X_test = pd.get_dummies(self.X_train, drop_first=True), pd.get_dummies(self.X_test, drop_first=True)
        self.xgb = XGBClassifier().fit(self.X_train, self.y_train)

        return classification_report(self.y_test, self.xgb.predict(self.X_test), output_dict=True)

p = pipeline("https://api.ignatius.io/api/report/export?reportId=ti2coyqg1&tableId=2363&exportType=csv&size=-1&tblName=1",
             "yvcUwr-aOJH1CoHduTIyP5M67KrsOfrqESs4tgahd4s")
app.include_router(p.router)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
