# downscale-satelliteLST
A python class for enhancing the spatial resolution of Land Surface Temperature (LST) raster data using statistical downscaling. The target resolution is determined by the LST predictors.

## Description
This class implements the typical workflow of a statistical downscaling scheme for enhancing the spatial resolution of satellite-derived Land Surface Temperatures (LST). It uses [GDAL](https://gdal.org/python/) to perform the resampling of the raster data and [scikit-learn](https://scikit-learn.org/stable/) for building the regression models data by combining two CART regressors ([ADAboost][sklern-adaboostRegr] & [Random Forest][sklern-RFregr]) and two linear regressors ([ElasticNET][sklern-elastnet] & [Ridge][sklern-rifge]) into an ensemble [VotingRegressor][sklern-voting].

Before using the class, the user **must**: (a) prepare and standarize the predictors; and (b) determine the best hyperparameters for each one of the employed `AdaBoost`, `RandomForest`, `ElasticNet` and `Ridge` regressors. The required hyperparameters are:

|  Regressor   |                            Required Hyperparameters                           |
|--------------|:-----------------------------------------------------------------------------:|
| AdaBoost     | [loss, n_estimators][sklern-adaboostRegr]                                     |
| RandomForest | [max_depth, n_estimators, min_samples_split, min_samples_leaf][sklern-RFregr] | 
| ElasticNet   | [l1_ratio, n_alphas, cv][sklern-elastnet]                                     |
| Ridge        | [alpha][sklern-rifge]                                                           |
	
[sklern-adaboostRegr]: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
[sklern-RFregr]:   https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
[sklern-elastnet]: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html
[sklern-rifge]:  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ridge_regression.html
[sklern-voting]: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html

### Input Data:
1. **LST**: A single raster dataset with one or more bands, where each band is a LST array.
2. **Predictors**: A single raster dataset with one or more bands, where each band is a predictor. Each band of the predictors should be standardized, i.e. centered over zero and with a variance of one.

The class does **not** require the two raster datasets to have the exact same SRS or Bounding  Box. The only requirement is the predictors to be **within** the bounds of the LST. It is very **important** however, that the projection and the geoTranformation coefficients of each raster to be correctly defined. If any of them is missing then the downscaling fuction will raise an error and stop. 

### Checks for downscaling the LST data:
If a LST band misses more than 40% of its pixels, then this band is discarded and no model is built. In addition, if a model achieves a R^2 that is lower than 0.5, it is also discarded. These two thresholds can be changed using the the setters 'SetMissingPxlsThreshold()' and 'SetR2Threshold()', respectively.

### Output:
A dictinary with the Downscaled LST (DLST) data of all the non-discarded models. The **spatial resolution** and the **SRS** of the output data will be that of the predictors.

### Things to keep in mind:
- It is recommended the datatype of the the LST and the predictors data to be **float32**.
- If the LST or the predictors contain any **water bodies** or **clouds**, then these pixels should be **NoData**.
- The class builds a **"global" regression model** for each LST band. Hence, it should be used with data that cover an area of **limited extent**, e.g. a city with its surroundings.
- The algorithm will generate data also for the cloud-covered areas. Handle them with caution.




## Usage
```python
from osgeo import gdal
from DownscaleSatelliteLST import DownscaledLST

# Make an instance of the class
data = DownscaledLST(
        LST=gdal.Open("inputLST.tif"),				 
        predictors=gdal.Open("LSTpredictors.tif"),   
        LST_noDataVal=-1000,						
        predictors_noDataVal=-1000,		
        workdir="./DLST_save_folder",
        )

# The class uses four regressors from scikit-learn and combines them
# into an ensemble VotingRegressor. Before applying the downscaling,
# it is mandatory to specify the regression parameters presented below
# using the following setters:
data.SetAdaBoostParams(loss="exponential", n_estimators=70)
data.SetRandomForestParams(max_depth=9, n_estimators=50, min_samples_split=2, min_samples_leaf=1)
data.SetElasticNetParams(l1_ratio=0.1, n_alphas=50, cv=5)
data.SetRidgeRegrParams(alpha=1.0)

# Update the R^2 threshold for discarding a model (the default value is 0.5)
data.SetR2Threshold(0.6)

# Downscale the LST data and apply the residual correction.
DLST = data.ApplyDownscaling(residual_corr=True)

# Get a list with LST bands that have been downcaled.
# LST bands that miss more than 40% of their pixels 
# and regression models that achieve a R^2 below the
# R2-threshold are discarded.
bands = data.GetDLSTBandIndices(indexing_from_1=False)

# Save in workdir a report with the
# scores of all the non-discarded models
data.GenerateReport()

# Save in workdir the DLST data as a compressed Geotiff file
data.SaveDLSTasGeotiff(savename="DLST.tif")
```


## To Do
- Add a class for preparing the predictors.
- Implement a function for finding the best parameters for each regressor
- Add unittests

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

If you use this class please cite:

    @phdthesis{Sismanidis2018,
        author = {Sismanidis, Panagiotis},
        pages  = {154},
        school = {National Technical University of Athens},
        title  = {{Applying Computational Methods for Processing Thermal Satellite Images of Urban Areas}},
        type   = {PhD Dissertation},
        year   = {2018}
    }