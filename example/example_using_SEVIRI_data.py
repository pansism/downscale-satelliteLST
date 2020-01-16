from osgeo import gdal
from DownscaleSatelliteLST import DownscaledLST

# Raster filenames
lst_fname = "LSALST_20180819_Athens_15min.tif"
predictors_fname = "LST_predictors_100m.tif"


def main():

    # Read the raster data as gdal datasets
    Seviri_LST = gdal.Open(lst_fname)
    predictors_100m = gdal.Open(predictors_fname)

    # Make an instance of the downscaling class
    data = DownscaledLST(
            LST=Seviri_LST,
            predictors=predictors_100m,
            LST_noDataVal=-80,
            predictors_noDataVal=-1000,
            workdir="./Results",
        )

    # Manually set the regression hyperparameters
    data.SetAdaBoostParams(loss="exponential", n_estimators=70)
    data.SetRandomForestParams(max_depth=9, n_estimators=50, min_samples_split=2, min_samples_leaf=1)
    data.SetElasticNetParams(l1_ratio=0.1, n_alphas=50, cv=5)
    data.SetRidgeRegrParams(alpha=1.0)

    # Set the R2 threshold for discarding
    # poorly-performing regression models.
    data.SetR2Threshold(0.5)

    # Apply the Downscaling
    data.ApplyDownscaling(residual_corr=True)

    # Save the dowscaled data as a geotiff file
    data.SaveDLSTasGeotiff(savename="downscaled_LST.tif")
    
    # Save the regression metrics in a txt file
    data.GenerateReport()

    # Get the indices of the non-discarded downscaled data
    print("LST bands that have been downscaled:")
    print(data.GetDLSTBandIndices(indexing_from_1=False))

if __name__ == "__main__":
    main()

