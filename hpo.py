import ray
from ray import air, tune
from ray.tune.search.bayesopt import BayesOptSearch
from cnn_mcwiliams import train_model
import torch
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch


ray.init()

space = { 
          "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
          "batch_size_train": tune.grid_search([4, 8, 16]),
          "learning_rate": tune.grid_search([1e-4, 5e-5, 1e-5]),
          "num_classes": 1,
          "num_epochs":200,
          "p_data":tune.grid_search([2000]),
}
my_point_to_evaluate = { 
          "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
          "batch_size_train": 16, 
          "learning_rate": 5e-5,
          "num_classes": 1,
          "p_data":2000,
}
'''
bayesopt = BayesOptSearch(
    space=space,
    metric="metric",
    mode="min",
    points_to_evaluate=my_point_to_evaluate,
    random_search_steps=10,
)
'''
my_tune_config = tune.TuneConfig(
    metric ="metric",
    mode="min",
    #search_alg=bayesopt,
    num_samples=1,
    max_concurrent_trials=1,
)

my_run_config = air.RunConfig(
    name="cnn_mcwiliams_hpo",
    local_dir = "/media/volume/sdc/cgft/test_for_online/cnn_mcwiliams/hpo_results",
    verbose=0,

)
trainable_with_gpu = tune.with_resources(train_model, {"gpu": 1})
tuner = tune.Tuner(
        trainable_with_gpu,
        param_space=space,
        tune_config=my_tune_config,
        run_config=my_run_config,
        
    )

results = tuner.fit()

#print("Best config: ", analysis.get_best_config(metric="mean_accuracy"))
print("Best hyperparameters found were: ", results.get_best_result().config)

