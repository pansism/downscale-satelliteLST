
if __name__ == "__main__":

    import platform
    from osgeo import gdal
    from DownscaleSatelliteLST import DownscaledLST

    if platform.system() == "Linux" or platform.system() == "Darwin":  # Darwin is Mac
        #predictors_fname = "/mnt/e/Work/MyCodeRepository/sandbox/lst-downscaling_forAGU/example/eco_20180821T132843pca_pred_stack.tif" #eco_20180821T132843pca_pred_stack_withAspGHSL.tif" 
        #lst_fname = "/mnt/e/Work/MyCodeRepository/sandbox/lst-downscaling_forAGU/example/lsalst_500scenes.tif" #lsalst_2018_1453scenes.tif"
        #savedir = "/mnt/e/Work/Projects/NASA UrbClim/GOES16 082018 LA data/Downscaling Results/AthensReRun_4ParamDistrs/500_scenes_rf+sgd"
        predictors_fname = "/mnt/e/Work/Projects/NASA UrbClim/GOES16 082018 LA data/data/predictors entire scene/funion_Smooth_v4.2.tif" 
        lst_fname = "/mnt/e/Work/Projects/NASA UrbClim/GOES16 082018 LA data/data/GOES LST/GOESLST_LA_201808082130.tif"
        savedir = "/mnt/e/Work/Projects/NASA UrbClim/GOES16 082018 LA data/Downscaling Results/Test_5/funion_Smooth_v4.2_corr"

    elif platform.system() == "Windows":
        predictors_fname = r"E:\Work\Projects\NASA UrbClim\GOES16 082018 LA data\Downscaling Results\Test_1\LA_predictors.tif"
        lst_fname= r"E:\Work\Projects\NASA UrbClim\GOES16 082018 LA data\Downscaling Results\Test 7/GOESLST_LA_201808082130_v3.tif"
        savedir = (
            r"C:\Users\User\Desktop\NEW_CODE"
        )
    else:
        print("Unable to determine which OS you are using")

    predictors=gdal.Open(predictors_fname).ReadAsArray()

    p = DownscaledLST(
        LST=gdal.Open(lst_fname),
        predictors=gdal.Open(predictors_fname),
        LST_noDataVal=-1000,
        predictors_noDataVal=-1000,
        workdir=savedir,
    )
    
    p.SetR2Threshold(0.10)
    p.SetNumberOfJobs(4)
    p.SetRandomSearchNumber(60)
    
    
    data = p.ApplyDownscaling(residual_corr=True)
    p.GenerateReport()
    print(p.GetDLSTBandIndices(indexing_from_1=True))
    p.SaveDLSTasGeotiff(savename="LAT_Sep09_16_DLST.tif")
