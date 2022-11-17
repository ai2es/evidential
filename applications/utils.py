#from keras import backend as K
import tensorflow as tf


def build_model(input_size, 
                hidden_size, 
                num_hidden_layers, 
                output_size, 
                activation, 
                dropout_rate):
    
    model = tf.keras.models.Sequential()
        
    if activation == 'leaky':
        model.add(tf.keras.layers.Dense(input_size))
        model.add(tf.keras.layers.LeakyReLU())
        
        for i in range(num_hidden_layers):
            if num_hidden_layers == 1:
                model.add(tf.keras.layers.Dense(hidden_size))
                model.add(tf.keras.layers.LeakyReLU())
            else:
                model.add(tf.keras.layers.Dense(hidden_size))
                model.add(tf.keras.layers.LeakyReLU())
                model.add(tf.keras.layers.Dropout(dropout_rate))
    else:
        model.add(tf.keras.layers.Dense(input_size, activation=activation))
        
        for i in range(num_hidden_layers):
            if num_hidden_layers == 1:
                model.add(tf.keras.layers.Dense(hidden_size, activation=activation))
            else:
                model.add(tf.keras.layers.Dense(hidden_size, activation=activation))
                model.add(tf.keras.layers.Dropout(dropout_rate))
      
    model.add(tf.keras.layers.Dense(output_size, activation='linear'))
    
    return model


def calc_prob_uncertinty(outputs, num_classes = 4):
    evidence = tf.nn.relu(outputs)
    alpha = evidence + 1
    S = tf.keras.backend.sum(alpha, axis=1, keepdims=True)
    u = num_classes / S
    prob = alpha / S
    epistemic = prob * (1 - prob) / (S + 1)
    aleatoric = prob - prob**2 - epistemic
    return prob.numpy(), u.numpy(), aleatoric.numpy(), epistemic.numpy()


if __name__ == "__main__":
    
    import pandas as pd
    import joblib
    import yaml
    import sys 
    import os
    
    config = sys.argv[1]
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)
        
    features = conf['tempvars'] + conf['tempdewvars'] + conf['ugrdvars'] + conf['vgrdvars']
    outputs = conf['outputvars']
    
    
    ### Load model and weights
    mlp = build_model(len(features), 
                  conf["trainer"]["hidden_sizes"], 
                  conf["trainer"]["num_hidden_layers"], 
                  len(outputs), 
                  conf["trainer"]["activation"], 
                  conf["trainer"]["dropout_rate"])

    mlp.build((conf["trainer"]["batch_size"], len(features)))
    mlp.summary()
    
    mlp.load_weights(os.path.join(conf["save_loc"], "best.pt"))
    
    ### Load scaler
    with open(os.path.join(conf["save_loc"], "scalers.pkl"), "rb") as fid:
        scaler_x = joblib.load(fid)
        
    ### Predict with some data
    test_data = pd.read_csv(os.path.join(conf["save_loc"], "test_df.csv"))
    preds = mlp.predict(scaler_x.transform(test_data[features]))
    probs, evidential, aleatoric, epistemic = calc_prob_uncertinty(preds)