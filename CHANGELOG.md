# CHANGELOG

## v.1.1.1 - 5 July 2022

- `DownscaledLST()` should now be used with scikit-learn v.1.1.1.
- `requirements.txt` was updated and `environment.yml` was added.
- Removed unused imports. Fixed some typos.
- The methods `_LST_to_radiance()` and `_radiance_to_LST()` were added. The regression modelling is now done in the radiance space.

## v.1.1.0 - 5 March 2020

- The ensemble model is now derived using a stacked regressor instead of a voting regressor. The estimator used for the stacking is an elasticNET regressor with built-in cross-validation.
- The ensemble model is now built using a random forest, a ridge regressor and a SVM. The adaboost and elasticNET regressors used in v.1.0.0 have been dropped.
- The class now uses random search with cross validation to fine-tune the hyperparameters of the 3 regressors and thus does not require the user to estimate them beforehand. The default number of hyperparameter searches is 60 but it can be changed using the setter `set_num_searches`.
- The class now transforms the input predictors to follow a normal distribution using sklearn's QuantileTransformer. The number of bins is uses are equal to 50% of the length of y.
- The setters `SetAdaBoostParams`, `SetRandomForestParams`, `SetElasticNetParams` and `SetRidgeRegrParams` have been deleted.
- The methods `set_num_searches` for setting how many hyperparameter sets that will be tested and `set_num_jobs` for setting the maximum number of parallel jobs (default is 1) have been addded.
