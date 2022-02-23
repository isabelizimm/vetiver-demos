from vetiver import mock, VetiverModel, VetiverAPI

X, y = mock.get_mock_data()
model = mock.get_mock_model().fit(X, y)

v = VetiverModel(model = model, save_ptype= True, ptype_data=X,\
    model_name="my_model", versioned=None, description="A regression model for testing purposes")

app = VetiverAPI(v)
