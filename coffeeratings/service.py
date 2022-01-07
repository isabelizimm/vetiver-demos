from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import uvicorn
import sklearn

model = joblib.load("coffeelr.joblib")
app = FastAPI()

class Coffee(BaseModel):
    aroma: float
    flavor: float
    sweetness: float
    acidity: float
    body: float
    uniformity: float
    balance: float


@app.get("/")
def vetiver_intro():
    return {"message": "go to /rapidoc or /docs"}

@app.post("/predict")
async def prediction(pred_data: Coffee):

    data = pred_data.dict()
    data_in = [[data['aroma'], data['flavor'], data['sweetness'], data['acidity'], data['body'], data['uniformity'], data['balance']]]

    y = model.predict(data_in)

    return {'prediction': y[0]}

@app.get("/rapidoc", response_class=HTMLResponse, include_in_schema=False)
async def rapidoc():
    return f"""
        <!doctype html>
        <html>
            <head>
                <meta charset="utf-8">
                <script 
                    type="module" 
                    src="https://unpkg.com/rapidoc/dist/rapidoc-min.js"
                ></script>
            </head>
            <body>
                <rapi-doc spec-url="{app.openapi_url}"></rapi-doc>
            </body> 
        </html>
    """

uvicorn.run(app, host="127.0.0.1", port=8000)