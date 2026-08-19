"""模型注册表。"""
from .arima_model import ArimaModel
from .lstm_model import LstmModel
from .svr_model import SvrModel
from .bp_model import BpModel
from .hybrid_model import HybridModel

MODELS = {
    "ARIMA": ArimaModel,
    "LSTM": LstmModel,
    "SVR": SvrModel,
    "BP": BpModel,
    "ARIMA-LSTM": HybridModel,
}
