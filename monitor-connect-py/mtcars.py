import pins
import vetiver

from vetiver.data import mtcars
from vetiver import VetiverModel
from sklearn import linear_model
from rsconnect.api import RSConnectServer

import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

api_key = os.getenv("API_KEY")
rsc_url = os.getenv("RSC_URL")
connect_server = RSConnectServer(url=rsc_url, api_key=api_key)

board = pins.board_rsconnect(
    server_url="https://colorado.rstudio.com/rsc", allow_pickle_read=True, api_key=api_key
)

# # ## board rsconnect, pin already on rsconnect
# # vetiver.deploy_rsconnect(
# #     connect_server=connect_server,
# #     board=board,
# #     pin_name="isabel.zimmerman/iz-bike"
# # )
# ## board folder
# # board = pins.board_rsconnect(path = ".", allow_pickle_read=True)

# # cars_lm = linear_model.LinearRegression().fit(mtcars.drop(columns="mpg"), mtcars["mpg"])

v = VetiverModel.from_pin(board, "isabel.zimmerman/iz-cars")
v.model_name = "cars"
b2 = pins.board_folder(path=".", allow_pickle_read=True)
vetiver.vetiver_pin_write(b2, v)
# vetiver.vetiver_pin_write(board, v)
# vetiver.write_app(board=board, pin_name="isabel.zimmerman/iz-bike", overwrite=True)
# vetiver.deploy_rsconnect(
#     connect_server=connect_server,
#     board=board,
#     pin_name="isabel.zimmerman/iz-bike",
#     version="58519",
#     extra_files=["cat.txt"]
# )
# m = vetiver.VetiverAPI(v)
# m.run()
# from rsconnect.actions import deploy_python_fastapi
# deploy_python_fastapi(
#     connect_server=connect_server,
#     directory=".",
#     extra_files=None,
#     excludes=None,
#     entry_point = "app:api",
#     new=True,
#     app_id=None,
#     title="mtcars linear regression",
#     python=None,
#     conda_mode=False,
#     force_generate=False,
#     log_callback=None,
# )
# from vetiver.server import predict, vetiver_endpoint

# endpoint = vetiver_endpoint("http://127.0.0.1:8000/predict")
# h = { 'Authorization': f'Key {api_key}' }
# print(predict(data=mtcars.drop(columns="mpg"), endpoint=endpoint, headers=h))
