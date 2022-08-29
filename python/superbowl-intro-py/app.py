from vetiver import VetiverModel
import vetiver
import pins


b = pins.board_folder('superbowl_rf', allow_pickle_read=True)
v = VetiverModel.from_pin(b, 'superbowl_rf', version = '20220819T113336Z-fd402')

vetiver_api = vetiver.VetiverAPI(v)
vetiver_api.run()
