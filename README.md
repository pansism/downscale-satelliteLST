# downscale-satelliteLST
A python class for enhancing the spatial resolution of Land Surface Temperature (LST) raster data using statistical downscaling.
![](LST_vs_DLST_compr.gif)

## Description
This class implements the typical workflow of a statistical downscaling scheme for enhancing the spatial resolution of satellite-derived Land Surface Temperatures (LST). It uses [GDAL](https://gdal.org/python/) to perform the resampling of the raster data and [scikit-learn](https://scikit-learn.org/stable/) to build the regression models by [stacking](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html) a [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html?highlight=random%20forest#sklearn.ensemble.RandomForestRegressor) a [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html?highlight=ridge#sklearn.linear_model.Ridge) and a SVM(https://scikit-learn.org/stable/modules/classes.html?highlight=svm#module-sklearn.svm) into an ensemble regressor.

### Input Data:
1. **LST**: A single raster dataset with one or more bands (each band is a LST image).
2. **Predictors**: A single raster dataset with one or more bands (each band is a predictor).

The class does **not** require the two raster datasets to have the same SRS and Bounding  Box. The only requirement is the predictors to be **within** the bounds of the LST data.

### Class checks before downscaling the LST data:
If a LST band misses more than 40% of its pixels, then this band is discarded and no model is built. In addition, if a model achieves a R^2 that is lower than 0.5, it is also discarded. These two thresholds can be changed using the setters `SetMissingPxlsThreshold()` and `SetR2Threshold()`, respectively.

### Output:
A dictionary with the Downscaled LST (DLST) data of all the non-discarded models. The **spatial resolution** and the **SRS** of the output data is that of the predictors.

To save the DLST data as a raster dataset use the class method `SaveDLSTasGeotiff()` and provide a savename (the savepath is the workdir).

## Usage
```python
from osgeo import gdal
from DownscaleSatelliteLST import DownscaledLST     # Import the class

# Make an instance of the class
data = DownscaledLST(
        LST=gdal.Open("inputLST.tif"),				 
        predictors=gdal.Open("LSTpredictors.tif"),   
        LST_noDataVal=-1000,						
        predictors_noDataVal=-1000,		
        workdir="./DLST_save_folder",
        )

# Change the number of parallel jobs to 4 (the default value is 1; to use all cores use -1)
data.SetNumberOfJobs(4)

# Change the R^2 threshold for discarding a model (the default value is 0.5)
data.SetR2Threshold(0.6)

# Downscale the LST data and apply the residual correction.
DLST = data.ApplyDownscaling(residual_corr=True)

# Get a list with the LST bands that have been downscaled.
# LST bands that miss more than 40% of their pixels 
# and regression models that achieve a R^2 below the
# R2-threshold are discarded.
bands = data.GetDLSTBandIndices(indexing_from_1=False)

# Save a report with the scores of all the non-discarded models
# The report is saved in workdir
data.GenerateReport()

# Export the DLST data as a compressed Geotiff file
# The geotiff file is saved in workdir 
data.SaveDLSTasGeotiff(savename="DLST.tif")
```
For a working example check the folder [example](example).

## Things to keep in mind:
- The recommended datatype for the input rasters is **float**.
- If the LST or the predictors contain any **water bodies** or **clouds**, then these pixels should be **NoData**.
- All the NoData pixels should have the **same** NoData value.
- The class builds a **"global" regression model** for each LST band. Hence, it should be used with data that cover an area of **limited extent**, e.g. a city with its surroundings.
- If the predictors are gapless, the algorithm will generate DLST data and for the cloud-covered areas. Handle them with caution.

## To Do
- Add a class for preparing the predictors.
- Add unittests

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Reference
If you use this class please cite:

    @phdthesis{Sismanidis2018,
        author = {Sismanidis, Panagiotis},
        pages  = {154},
        school = {National Technical University of Athens},
        title  = {{Applying Computational Methods for Processing Thermal Satellite Images of Urban Areas}},
        type   = {PhD Dissertation},
        year   = {2018}
    }
