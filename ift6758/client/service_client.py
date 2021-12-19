import json
import requests
import pandas as pd
import logging
from ift6758.client.feature_lists import *
from ift6758.models.utils import load_data

logger = logging.getLogger(__name__)


class ServingClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 5000, features=None):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        if features is None:
            features = ["distance"]
        self.features = features

        self.model_registries_to_file_name = {
            '6-lgbm': ('6-LGBM.pkl', feature_list_lgbm),
            '5-2-grid-search-model': ('tuned_xgb_model.pkl', feature_list_xgb),
            '6-2-nn-tuned-model': ('tuned_nn_model.pkl', feature_list_nn),
            '6-4-stacked-trained-tuned-model': ('tuned_stacked_trained_model.pkl',feature_list_stack_trained),
            '3-3-angle-dist-logreg-model':('LogReg_dist_angle_model.pkl',feature_list_logreg)
        }

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.
        
        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """
        
        for column in self.features:
            if column not in X.columns:
                x[column] = np.zeros(x.shape[0])
        
        X = X[self.features] #need to make sure columns are in the same order as training time
        X = X.reset_index()
        r = requests.post(f"{self.base_url}/predict", json=X.to_json())
        result = pd.read_json(r.json())
        return result

    def logs(self) -> dict:
        """Get server logs"""

        r = requests.get(f"{self.base_url}/logs")
        return r.json()

    def download_registry_model(self, workspace: str, model: str, version: str) -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it. 
        See more here:
            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model
        
        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """

        assert model in self.model_registries_to_file_name.keys(), f'model name must be in ' \
                                                                   f'{self.model_registries_to_file_name.keys()} '
        model_file_name = self.model_registries_to_file_name[model][0]
        self.features = self.model_registries_to_file_name[model][1]
        request = {'workspace': workspace, 'registry_name': model, 'model_name': model_file_name, 'version': version}
        r = requests.post(f"{self.base_url}/download_registry_model", json=request)
        return r.json()

© 2021 GitHub, Inc.
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About
