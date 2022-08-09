from vetiver import VetiverModel
import vetiver
import pins

import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

api_key = os.getenv("API_KEY")
rsc_url = os.getenv("RSC_URL")

b = pins.board_rsconnect(
    server_url="https://colorado.rstudio.com/rsc", 
    allow_pickle_read=True, 
    api_key=api_key
)

v = VetiverModel.from_pin(b, 'isabel.zimmerman/superbowl_rf', version="57290")

api = vetiver.VetiverAPI(v)
api.run()
