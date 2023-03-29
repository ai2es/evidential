from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
import logging


logger = logging.getLogger(__name__)


def parse_preprocessor(string, seed = 1000):
    if string == "standard":
        return StandardScaler()
    elif string == "normalize":
        return MinMaxScaler((0, 1))
    elif string == "symmetric":
        return MinMaxScaler((-1, 1))
    elif string == "robust":
        return RobustScaler()
    elif string == "quantile":
        return QuantileTransformer(
            n_quantiles=1000, 
            random_state=seed, 
            output_distribution = "normal"
        )
    else:
        raise OSError(
            "Preprocessing type not recognized. Select from standard, normalize, symmetric, robust, or quantile"
        )


def load_preprocessing(conf, seed = 1000):
    if "preprocessing" not in conf["data"]:
        return False, False
    
    input_scaler = False
    output_scaler = False
    if "inputs" in conf["data"]["preprocessing"]:
        scaler = conf["data"]["preprocessing"]["inputs"]
        logger.info(f"Loading input preprocessing scaler(s): {scaler}")
        input_scaler = parse_preprocessor(
            scaler,
            seed = seed
        )
    if "outputs" in conf["data"]["preprocessing"]:
        scaler = conf["data"]["preprocessing"]["outputs"]
        logger.info(f"Loading output preprocessing scaler(s): {scaler}")
        output_scaler = parse_preprocessor(
            scaler,
            seed = seed
        )
    return input_scaler, output_scaler
    