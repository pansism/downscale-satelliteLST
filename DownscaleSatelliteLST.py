# -*- coding: utf-8 -*-
#!/usr/bin/env python3.6
"""
A class for enhancing the spatial resolution of satellite-derived Land Surface
Temperatures (LST) raster data using statistical downscaling. The target resolution
is determined by the LST predictors.

Usage Example:
    >>> from osgeo import gdal
    >>> from DownscaleSatelliteLST import DownscaledLST
    >>> data = DownscaledLST(
        LST=gdal.Open("inputLST.tif"),
        predictors=gdal.Open("LSTpredictors.tif"),
        LST_noDataVal=-1000,
        predictors_noDataVal=-1000,
        workdir="./DLST_save_folder",
        )
    >>> data.SetNumberOfJobs(2)
    >>> DLST = data.ApplyDownscaling(residual_corr=True)
        
    Downscaling started at:   05/03/2020, 12:00

    SETTINGS
    ========
    Residual Correction:           True
    R2-threshold:                  0.1
    Missing pxls threshold:        40.0%
    Train/test size split:         0.7/0.3
    Parallel jobs:                 2
    Hyperarameter tuning trials:   60
    
    Building the regression models.
    Processing band 0:
        Tuning the random forest hyperparameters...   Done [CV R2 score = 0.63]
        Tuning the ridge hyperparameters...           Done [CV R2 score = 0.48]
        Tuning the svr hyperparameters...             Done [CV R2 score = 0.59]
        The R2 score of the ensemble model is: 0.64   PASS

    Models that passed the checks: 1/1

    Downscaling the corresponding LST bands...
    Downscaling LST band 0:   [#########################] 100.00% 

    Downscaling completed in: 127.1 sec
    >>> type(DLST)
    dict
    >>> data.GetDLSTBandIndices(indexing_from_1=True)
    [1]
    >>> data.GenerateReport()
    Generating report...      DONE
    >>> data.SaveDLSTasGeotiff(savename="DLST.tif")
    Writing to GeoTiff...     DONE

*********************************************************************************************************

Version:         1.1.0
Release Date:    28 November 2019
Last Update:     5 March 2020

Author:          Panagiotis Sismanidis
Address:         National Observatory of Athens, Greece
e-mail:          panosis@noa.gr

If you use this class please cite:

    @phdthesis{Sismanidis2018,
        author = {Sismanidis, Panagiotis},
        pages  = {154},
        school = {National Technical University of Athens},
        title  = {{Applying Computational Methods for Processing Thermal Satellite Images of Urban Areas}},
        type   = {PhD Dissertation},
        year   = {2018}
    }

Enjoy!

This software is provided under the MIT license.

Copyright 2019-2020. Panagiotis Sismanidis

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import numpy.ma as ma
import concurrent.futures
import os
import sys
import sklearn
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, RidgeCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn import metrics
from datetime import datetime
from osgeo import gdal, gdal_array, gdalconst, osr
from scipy import stats
from distutils.version import StrictVersion


class DownscaledLST:
    def __init__(self, LST, predictors, LST_noDataVal, predictors_noDataVal, workdir):
        """A class for enhancing the spatial resolution of satellite LST.

        The spatial resolution and the SRS of the output Downscaled LST (DLST) will be
        that of the predictors. The class is based on GDAL and the two main inputs,
        i.e. the LST and the predictors, have to be GDAL.datasets with one or more
        bands. If the projection or the geoTranformation information, from any of the
        two datasets, is missing then the downscaling fuction will not run. The class
        does not require the two raster datasets to have the exact same SRS or Bounding
        Box. The only requirement is the predictors to be within the bounds of the LST.

        Arguments:
            LST {gdal.Dataset} -- A single LST raster with one or more bands
            predictors {gdal.Dataset} -- A single predictors raster  with one or more bands
            LST_noDataVal {ind or float} --  The noData value of the LST
            predictors_noDataVal {ind or float} --  The noData value of the predictors
            workdir {str} -- The working directory
        """
        
        for input_raster in [LST, predictors]:
            self._validate_input(input_raster)

        self.workdir = workdir
        if not os.path.exists(workdir):
            os.makedirs(workdir)

        self.predictors = predictors
        self.predictors_NDV = predictors_noDataVal
        self.LST_NDV = LST_noDataVal

        # Make an instance variable with the predictor's bounding box and SRS
        self._GetPredictorsBBox()

        # Clip the LST data to the predictor's BBox
        self.LST = self._WarpRaster(
            dst=LST,
            src=LST,
            dst_ndv=self.LST_NDV,
            src_ndv=self.LST_NDV,
            resampling="nearest",
        )

        # Resample the predictors to the LST grid
        self.upscaled_predictors = self._WarpRaster(
            dst=LST,
            src=self.predictors,
            dst_ndv=self.predictors_NDV,
            src_ndv=self.predictors_NDV,
            resampling="average",
        )

        # The settings for _BuildRegrModel() 
        self.regr_test_size = 0.3
        self.SEED = 123
        self.N_JOBS = 1
        self.N_RANDOM_SEARCHES = 60

        # Thresholds for discarding regression models / LST bands
        self.cloud_cover_threshold = 0.4
        self.R2_threshold = 0.5

        # The resampling method used in the residual correction
        self.supersampling_method = "cubspline"

        # The output Downscaled LST (DLST) and the model scores.
        self.DLST = {}
        self.model_scores = {}

        self.__version__ = "1.1.0"


    def ApplyDownscaling(self, residual_corr):
        """Enhance the spatial resolution of the LST data using statistical downscaling.

        This function starts by upscaling the given predictors to the LST data. Then, it uses
        the upscaled predictors, and for each LST band, it builds an ensemble regression
        model that describes their relationship. If a LST band misses more pixels than the
        predefined threshold, then this band is discarded and no model is built. In addition,
        if a model achieves a R^2 that is lower than the predifed threshold, it is also discarded.
        The default pxl- and R^2-thresholds are 40% and 0.5, respectively. To change them use
        the setters 'SetMissingPxlsThreshold()' and 'SetR2Threshold()', respectively. After building
        all the models, this function applies each ensemble model to the given predictors so as
        to retrieve the Downscaled LST (DLST) data. If the 'residual_corr' flag is set to 'True',
        the DLST residual correction is also applied at this stage.

        The spatial resolution and the SRS of the output DLST data is that of the predictors. To use this
        function, it is mandatory to first specify the required regression parameters using the setters:
        'SetAdaBoostParams()', 'SetRandomForestParams()', 'SetElasticNetParams()' and 'SetRidgeRegrParams()'.

        The class builds a "global" regression model for each LST band and hence it should be used with data
        that cover an area of limited extent, e.g. a city with its surroundings.

        Arguments:
            residual_corr {bool} -- Residual correction flag

        Returns:
            dict -- The Downscaled LST (DLST) data
        """
        start = datetime.now()

        assert (
            isinstance(residual_corr, bool) == True
        ), "The 'residual_corr' argument should be True or False."
        assert StrictVersion(sklearn.__version__) >= StrictVersion(
            "0.22.2"
        ), "Sklearn v.0.22.2 or greater is required."

        print(f"\n{'Downscaling started at:':<25} {start.strftime('%d/%m/%Y, %H:%M')}")
        print("\nSETTINGS")
        print("="*8)
        print(f"{'Residual Correction:':<30} {residual_corr}")
        print(f"{'R2-threshold:':<30} {self.R2_threshold}")
        print(f"{'Missing pxls threshold:':<30} {self.cloud_cover_threshold*100}%")
        print(f"{'Train/test size split:':<30} {1-self.regr_test_size}/{self.regr_test_size}")
        print(f"{'Parallel jobs:':<30} {self.N_JOBS}")
        print(f"{'Hyperarameter tuning trials:':<30} {self.N_RANDOM_SEARCHES}")

        LST = self._GetMskdArray(self.LST, self.LST_NDV)
        predictors = self._GetMskdArray(self.predictors, self.predictors_NDV)
        upscaled_predictors = self._GetMskdArray(
            self.upscaled_predictors, self.predictors_NDV
        )

        # Use the upsaled predictors to estimate how many the non-nan LST pxls are.
        pxl_total = np.count_nonzero(upscaled_predictors.mask.any(axis=0) == False)

        print(f"\nBuilding the regression models.")
        models = {}
        for i, LST_band in enumerate(LST):
            combined_nanmask = np.logical_or(LST_band.mask, upscaled_predictors.mask)
            
            y = LST_band[combined_nanmask.any(axis=0) == False]
            X = upscaled_predictors[:, combined_nanmask.any(axis=0) == False].T  # Coarse resolution predictors

            clear_sky_pxl_perc = len(y) / pxl_total  # this is a rough estimate

            if clear_sky_pxl_perc >= self.cloud_cover_threshold:
                print(f"  Processing band {i}:")
                normal_transformer = QuantileTransformer(len(y)//2, "normal", random_state=self.SEED).fit(X)
                model, metrics = self._BuildRegrModel(y, normal_transformer.transform(X))
                R2 = metrics[0]
                print(f"{f'    The R2 score of the ensemble model is: {R2:0.2f}':<50}", end="")

                if R2 >= self.R2_threshold:
                    models[i] = model
                    self.model_scores[i] = metrics
                    print("PASS")
                else:
                    print("FAIL")
            else:
                print(f"    Band {i} failed the missing pixels test - Band DISCARDED.")

        if bool(models) == False:
            raise SystemExit("All the models failed the R2 and pixel-% tests.")

        print(f"\n{'Models that passed the checks:':<25} {len(models)}/{self.LST.RasterCount}")
        print(f"\nDownscaling the corresponding LST bands...")

        X = predictors[:, predictors.mask.any(axis=0) == False].T  # Fine resolution predictors
        X_idx = np.argwhere(predictors.mask.any(axis=0) == False)

        X_normal = normal_transformer.transform(X).T

        # If X_normal is larger than MAX_ELEMENTS, split it into chuncks to
        # avoid any memory overflow when applying predict().
        split_X_into_chuncks = False
        MAX_ELEMENTS = 250000
        if X_normal.shape[1] > MAX_ELEMENTS:
            splits = X_normal.shape[1] // MAX_ELEMENTS
            if splits > 1:
                X_normal = np.array_split(X_normal.T, splits, axis=0)
                X_idx = np.array_split(X_idx.T, splits, axis=1)
                split_X_into_chuncks = True

        for i, (band, model) in enumerate(models.items()):

            DLST_array = np.zeros(
                shape=(self.predictors.RasterYSize, self.predictors.RasterXSize),
                dtype="float32",
            )

            if split_X_into_chuncks == True:
                for split, X_subst in enumerate(X_normal):
                    self._progressbar(splits-1, split, f"Downscaling LST band {band}:")
                    DLST = model.predict(X_subst)
                    DLST_array[tuple(X_idx[split])] = DLST

            else:
                DLST = model.predict(X_normal.T)
                DLST_array[tuple(X_idx.T)] = DLST
                self._progressbar(1, 1, f"Downscaling LST band {band}:")

            if residual_corr == True:
                residuals = self._CalcResiduals(DLST_array, LST[band])
                DLST_array += residuals

            DLST_array[DLST_array == 0] = self.LST_NDV
            self.DLST[band] = DLST_array

        elapsed_time = (datetime.now() - start).total_seconds()
        print(f"\n{'Downscaling completed in:':<25} {elapsed_time:.01f} sec")

        return self.DLST


    def GetDLSTBandIndices(self, indexing_from_1):
        """Get a list with the LST bands that have been downscaled."""
        assert len(self.DLST) > 0, "Apply the Downscaling first."

        if indexing_from_1 == True:
            return [idx + 1 for idx in self.DLST.keys()]
        else:
            return list(self.DLST.keys())


    def SaveDLSTasGeotiff(self, savename):
        """Write the DLST to a compressed Geotiff file.

        Arguments:
            savename {str} -- The name of the Geotiff file
        """
        assert len(self.DLST) > 0, "Apply the Downscaling first."

        driver = gdal.GetDriverByName("GTiff")
        gtiff = driver.Create(
            os.path.join(self.workdir, savename),
            xsize=self.predictors.RasterXSize,
            ysize=self.predictors.RasterYSize,
            bands=len(self.DLST),
            eType=gdal.GDT_Float32,
            options=["COMPRESS=DEFLATE", "BIGTIFF=IF_NEEDED"],
        )
        gtiff.SetGeoTransform(self.predictors.GetGeoTransform())
        gtiff.SetProjection(self.predictors.GetProjection())

        print(f"{'Writing to GeoTiff...':<25}", end=" ")
        try:
            for i, sceneID in enumerate(self.DLST.keys()):
                band = gtiff.GetRasterBand(i + 1)
                band.WriteArray(self.DLST[sceneID].astype(np.float32))
                band.SetNoDataValue(self.LST_NDV)
        except:
            raise ValueError("Failed to write DLST data to GeoTiff.")

        gtiff.FlushCache()
        print("Done")


    def GenerateReport(self):
        """Save a report with the scores of all the non-discarded models."""

        assert len(self.DLST) > 0, "Apply the Downscaling first."

        header = "Performance metrics for all the non-discarded models."
        table_labels = {
            "Band": "The downscaled LST band",
            "R2": "Coefficient of Determination",
            "explVar": "Explained Variance score",
            "MaxRes": "Maximum Residual Error",
            "MAE": "Mean Absolute Error",
            "MedAE": "Median Absolute Error",
            "MSE": "Mean Squared Error",
        }

        print(f"{'Generating report...':<25}", end=" ")

        with open(os.path.join(self.workdir, "Model_Scores.txt"), "w") as report:

            # Print the table legend
            print("", file=report)
            for label, label_descr in table_labels.items():
                print(f"{label:<10}{label_descr}", file=report)

            # Print the table header
            print("", file=report)
            print(f"{header:^66}", file=report)
            print("=" * 66, file=report)
            print(
                f"".join(f"{label:<10}" for label in table_labels.keys()), file=report
            )
            print("=" * 66, file=report)

            # Fill the table
            for band in self.model_scores.keys():
                score_row = "".join(
                    f"{score:<10.02f}" for score in self.model_scores[band]
                )
                print(f"{band:<10}{score_row}", file=report)
            print("=" * 66, file=report)

        print("Done")


    def _GetMskdArray(self, raster, ndv):
        """Read a raster file as a masked array and change the NDVs to NaNs."""
        array = raster.ReadAsArray().astype(np.float32)
        array[array == ndv] = np.nan
        array = ma.masked_invalid(array)

        if raster.RasterCount == 1:
            array = np.expand_dims(array, axis=0)

        return array


    def _GetPredictorsBBox(self):
        """Get the bounding box coordinates and SRS of the fine resolution predictors."""
        geoTF = self.predictors.GetGeoTransform()
        MinX = geoTF[0]
        MinY = geoTF[3] + geoTF[5] * self.predictors.RasterYSize
        MaxX = geoTF[0] + geoTF[1] * self.predictors.RasterXSize
        MaxY = geoTF[3]

        proj = self.predictors.GetProjection()
        SRS = osr.SpatialReference(wkt=proj)

        self.BBox = {"coords": (MinX, MinY, MaxX, MaxY), "SRS": SRS}


    def _WarpRaster(self, dst, dst_ndv, src, src_ndv, resampling):
        """For the the predictors' BBox, warp the src raster to match the dst raster.

        For the fine resolution predictor's bounding box, use GDAL's warp function
        so as to match the source raster (src) to the target raster (dst).
        The warped raster will be saved as a GDAl's virtual raster dataset (VRT) in a
        folder called 'Intermediate VRTs' that will be created in the workdir.

        Arguments:
            dst {gdal.Dataset} -- The target raster
            dst_ndv {int or float} -- The dst NoData value
            src {gdal.Dataset} -- The source raster
            src_ndv {int or float} -- The src NoData value
            resampling {string} -- The resampling method (average, cubspling or nearest)

        Returns:
            gdal.Dataset -- The warped data as a virtual raster (VRT)
        """
        vrt_savedir = os.path.join(self.workdir, "Intermediate VRTs")
        if not os.path.exists(vrt_savedir):
            os.makedirs(vrt_savedir)

        src_fname = os.path.basename(src.GetDescription())
        vrt_fname = os.path.splitext(src_fname)[0] + "_WARPED_" + resampling + ".vrt"
        savepath = os.path.join(vrt_savedir, vrt_fname)

        resampling_methods = {"average": 5, "cubspline": 3, "nearest": 0}

        warp_options = gdal.WarpOptions(
            format="VRT",
            outputBounds=self.BBox["coords"],
            outputBoundsSRS=self.BBox["SRS"],
            srcSRS=osr.SpatialReference(wkt=src.GetProjection()),
            dstSRS=osr.SpatialReference(wkt=dst.GetProjection()),
            xRes=dst.GetGeoTransform()[1],
            yRes=abs(dst.GetGeoTransform()[5]),
            srcNodata=src_ndv,
            dstNodata=dst_ndv,
            resampleAlg=resampling_methods[resampling],
        )
        return gdal.Warp(savepath, src, options=warp_options)


    def _BuildRegrModel(self, y, X):
        """Train an ensemble regression model and assess its performance.

        Start by splitting the y and X to train and test samples. Then create three regressors,
        namely a Random Forest, a Ridge and a SVM regressor and tune their hyperparameters using
        random search with cross validation. After updating their hyperparamters stack the three
        regressors using an ElasticNET linear regression model and fit the ensemble model to the 
        train sample. Finally, calculate its performance using the test sample and return
        both the ensemble model and the calculated metrics.

        Arguments:
            y {numpy.ndarray} -- The response variable (i.e. the LST data)
            X {numpy.ndarray} -- The explanatory variables (i.e. the LST predictors)

        Returns:
            sklearn.ensemble._stacking.StackingRegressor -- The ensemble regression model
            tuple -- A tuple with the regression performance metrics
        """

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.regr_test_size, random_state=self.SEED
        )

        regressors = [
            ("random forest", RandomForestRegressor(random_state=self.SEED, n_jobs=self.N_JOBS)),
            ("ridge", Ridge(random_state=self.SEED)),
            ("svr", SVR()),
        ]

        hyperparam_distributions = {
            "random forest": {
                "max_depth": stats.randint(5, 100),
                "n_estimators": stats.randint(30, 800),
                "min_samples_leaf": stats.randint(2, 20),
                "min_samples_split": stats.randint(2, 50),
            },
            "svr": {
                "kernel": ["rbf", "poly", "sigmoid", "linear"],
                "degree":stats.randint(2, 7),
                "epsilon": stats.uniform(0.05, 5.0),
                "C": stats.uniform(0.0, 25.0),
            },
            "ridge": {"alpha": stats.uniform(0.0001, 1.0)},
        }

        for name, regressor in regressors:
            print(f"{f'    Tuning the {name} hyperparameters...':<50}", end="")
            hyperparam_candidates = RandomizedSearchCV(
                regressor,
                param_distributions=[hyperparam_distributions[name]],
                scoring="r2",
                random_state=self.SEED,
                n_jobs=self.N_JOBS,
                n_iter=self.N_RANDOM_SEARCHES,
                verbose=0,
            ).fit(X_train, y_train)
            print(f"Done [CV R2 score = {hyperparam_candidates.best_score_:0.2f}]")
            regressor.set_params(**hyperparam_candidates.best_params_)

        ensemble_regressor = StackingRegressor(
            regressors,
            final_estimator=ElasticNetCV(
                l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
                cv=10,
                n_jobs=self.N_JOBS,
                random_state=self.SEED,
                ),
            n_jobs=self.N_JOBS,
            passthrough=True,
        )

        try:
            ensemble_regressor.fit(X_train, y_train)
        except ValueError as err:
            raise ValueError(
                f"Error in _BuildRegrModel: Unable to fit ensemble regression model. {err}"
            )

        # Assess the model performance using the test data
        y_pred = ensemble_regressor.predict(X_test)

        #y_pred = regressors[1][1].predict(X_test)
        regr_metrics = (
            metrics.r2_score(y_test, y_pred),
            metrics.explained_variance_score(y_test, y_pred),
            metrics.max_error(y_test, y_pred),
            metrics.mean_absolute_error(y_test, y_pred),
            metrics.mean_squared_error(y_test, y_pred),
            metrics.median_absolute_error(y_test, y_pred),
        )

        return ensemble_regressor, regr_metrics


    def _CalcResiduals(self, DLST, LST):
        """Caclculate the residuals between the corresponding DLST and LST data.

        This function applies the HUTS residual correction (Dominguez et al. 2011).
        It starts by resampling the input DLST data array to the LST coarse resolution
        grid. It then subtracts the input LST and DLST arrays and calculates the
        corresponding LST-DLST residuals. Finally, it supersamples the derived residuals
        to the fine resolution grid of the predictors using using cubic spline interpolation,
        so as to avoid any boxing effects. To change the utilised resampling method use the
        'SetSupersamplingMthd()' method before running the 'ApplyDownscaling()' method.

        Arguments:
            DLST {numpy.ndarray} -- The uncorrected DLST data array
            LST {numpy.ndarray} -- The corresponding LST data array

        Returns:
            numpy.ndarray -- The supersampled LST-DLST residuals array
        """

        # Make a mask with the DLST noData pixels
        mask = DLST == 0

        # Resample the input DLST array to the LST grid
        DLST = gdal_array.OpenArray(DLST)
        DLST.SetGeoTransform(self.predictors.GetGeoTransform())
        DLST.SetProjection(self.predictors.GetProjection())
        DLST.SetDescription("ResidCorr_DLSTarray")
        upscaled_DLST_VRT = self._WarpRaster(
            dst=self.LST,
            src=DLST,
            dst_ndv=self.LST_NDV,
            src_ndv=0,
            resampling="average",
        )
        upscaled_DLST = upscaled_DLST_VRT.ReadAsArray()
        upscaled_DLST[upscaled_DLST == self.LST_NDV] = np.nan

        # Calculate the LST-DLST residuals
        residuals = np.subtract(LST, upscaled_DLST)
        residuals[np.isnan(residuals)] = 0

        # Supersample the residuals to the DLST grid
        residuals = gdal_array.OpenArray(residuals)
        residuals.SetGeoTransform(upscaled_DLST_VRT.GetGeoTransform())
        residuals.SetProjection(upscaled_DLST_VRT.GetProjection())
        residuals.SetDescription("ResidCorr_RESIDarray")
        supersampled_residuals_VRT = self._WarpRaster(
            dst=self.predictors,
            src=residuals,
            dst_ndv=0,
            src_ndv=np.nan,
            resampling=self.supersampling_method,
        )
        supersampled_residuals = supersampled_residuals_VRT.ReadAsArray()
        supersampled_residuals[mask == True] = 0

        return supersampled_residuals


    def _progressbar(self, total, iteration, message):
        """Displays a console progress bar."""
        barLength, status = 25, ""

        progress = float(iteration) / float(total)
        if progress >= 1.0:
            progress, status = 1, "\r\n"

        block = int(round(barLength * progress))

        text = "\r{:<25} [{}] {:.02f}% {}".format(
            message,
            "#" * block + "-" * (barLength - block),
            round(progress * 100, 0),
            status,
        )
        sys.stdout.write(text)
        sys.stdout.flush()


    def _validate_input(self, raster):
        '''Check if the input raster is a gdal dataset with SRS attributes.'''
        
        if isinstance(raster, gdal.Dataset) == False:
            raise TypeError("The input raster must be a gdal.Dataset with one or more bands.")
        if bool(raster.GetProjection()) == False:
            raise ValueError("The raster's proj definition is missing.")
        if raster.GetGeoTransform() == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
            raise ValueError("The raster's GeoTranformation coefficients are missing.")


    def SetNumberOfJobs(self, n_jobs):
        """Set scikit-learn's random number generation control seed."""
        self.N_Jobs = n_jobs


    def SetTestSize4Regr(self, test_size):
        """Set the proportion of the regression data to be used for testing."""
        self.regr_test_size = 0.2


    def SetRandomSeed(self, seed):
        """Set scikit-learn's random number generation control seed."""
        self.SEED = seed


    def SetRandomSearchNumber(self, n_random_searches):
        """Set how many hyperparameter candidates will be tested."""
        self.N_RANDOM_SEARCHES = n_random_searches
        if n_random_searches < 60:
            print("It is advisable to test at least 60 candidates.")


    def SetR2Threshold(self, threshold):
        """Set the R2 theshold below which a regression model will be discarded."""
        if threshold > 1 or threshold < 0:
            raise ValueError("R2 threshold should range between 0 and 1")

        self.R2_threshold = threshold


    def SetMissingPxlsThreshold(self, percentage):
        """Set the percentage of missing pixels below which a LST scene will be discarded."""
        if percentage > 100 or percentage < 0:
            raise ValueError(
                "The missing pixels threshold should range between 0% and 100%"
            )
        self.cloud_cover_threshold = percentage / 100


    def SetSupersamplingMthd(self, method):
        """Set the residual correction's resampling method."""
        methods = ["average", "nearest", "cubspline"]

        if method in methods:
            self.supersampling_method = method
        else:
            raise ValueError(f"Valid supersampling methods: {methods}.")
