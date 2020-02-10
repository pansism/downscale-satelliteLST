import platform
from osgeo import gdal
from DownscaleSatelliteLST import DownscaledLST

if platform.system() == "Linux" or platform.system() == "Darwin":  # Darwin is Mac
    #lst_fname = "/Users/panosis/Dropbox/MyCodeRepository/sandbox/Larissa_LST_UTM34N_120m_clp_v3.tif"
    #predictors_fname = "/Users/panosis/Dropbox/MyCodeRepository/sandbox/Larissa_B5_UTM34N_30m_clp_v2.tif"
    lst_fname = "/Users/panosis/Dropbox/MyCodeRepository/lst-downscaling/example/lsalst_2018_0809.tif"
    predictors_fname = "/Users/panosis/Dropbox/MyCodeRepository/lst-downscaling/example/eco_20180821T132843pca_pred_stack.tif"
    #lst_fname = "/Users/panosis/Desktop/DLST_tests/lC8_thess_500m.tif"
    #predictors_fname = "/Users/panosis/Desktop/DLST_tests/LC08_184032_20190508_LST.LST.tif"
    savedir = "/Users/panosis/Desktop/larissa_test5"

elif platform.system() == "Windows":
    lst_fname = r"C:\Users\User\Dropbox\MyCodeRepository\lst-downscaling\example\LSALST_Athens_Aug17_21.tif"
    predictors_fname = r"C:\Users\User\Dropbox\MyCodeRepository\lst-downscaling\example\eco_20180821T132843pca_pred_stack.tif"
    savedir = (
        r"C:\Users\User\Desktop\AGU19_presentation\LSALST_Athens_Aug17_21_with_resid"
    )
else:
    print("Unable to determine which OS you are using")

def main():

    Seviri_LST = gdal.Open(lst_fname)
    predictors_100m = gdal.Open(predictors_fname)
    
    data = DownscaledLST(
            LST=Seviri_LST, 
            predictors=predictors_100m,
            LST_noDataVal=-1000,
            predictors_noDataVal=-1000,
            workdir="./DLST_Example",
        )
        
    data.SetAdaBoostParams(loss="exponential", n_estimators=70)
    data.SetRandomForestParams(max_depth=9, n_estimators=50, min_samples_split=2, min_samples_leaf=1)
    data.SetElasticNetParams(l1_ratio=0.1, n_alphas=50, cv=5)
    data.SetRidgeRegrParams(alpha=1.0)
    
    # Increase the R2 threshold for discarding 
    # poorly-performing regression models.
    data.SetR2Threshold(0.4)

    data.ApplyDownscaling(residual_corr=False)
    data.SaveDLSTasGeotiff(savename="Athens_Seviri_DLST.tif")
    data.GenerateReport()

    print("LST bands that have been downscaled:")
    print(data.GetDLSTBandIndices(indexing_from_1=False))

if __name__ == "__main__":
    main()

