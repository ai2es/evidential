from evml.pit import pit_histogram, pit_deviation
import numpy as np

def test_pit_histogram_ensemble():
    n_samples = 10000
    n_members = 20
    uniform_ensemble = np.random.random(size=(n_samples, n_members))
    y_true_ens = uniform_ensemble[:, 0]
    y_ture_high = np.ones(n_samples) * 2
    y_true_low = np.ones(n_samples) * -1
    dev_ens = pit_deviation(y_true_ens, uniform_ensemble, pred_type="ensemble")
    dev_high = pit_deviation(y_true_high, uniform_ensemble, pred_type="ensemble")
    dev_low = pit_deviation(y_true_low, uniform_ensemble, pred_type="ensemble")
    assert dev_ens < dev_high and dev_ens < dev_low, f"Dev ens ({dev_ens:0.3f}) too high"
    assert dev_high == dev_low , f"Dev high ({dev_high:0.3f})not equal to dev low ({dev_low:0.3f})"
    return
