# ASD adaptive behavior subtypes from NDAR dataset

We present here how to run the code to enable the stratification of Vineland scores from the NDAR database and longitudinal analysis of UCSD database.

### Requirements
please create a new python virtual environment with the following specifics
```
Python = 3.8.1
``` 
Activate the environment in the folder and then
Install all the libraries with:
```
pip install -r requirements.txt
```
to run the R code, please use
```
R >=4.0
```
#### why do I have to do that?
beacuse numpy/sklearn/pandas need a specific version (up to the 3rd decimal) in order to replicate the exact same results. Lots of seeds are sets within this pipeline and they are toll-version- specific. that is set.seed(1) in on version of numpy is not the same set.seed(1) of another verision. so, in order to avoid this problem, please create another environment in which execute this pipeline. Be aware that the there might be some small differences in the apply UMAP (preprocessing steps before running reval) that might be due to the specific type of computer you are using for the analysis, which influence the way this UMAP library works.


### Pipeline 
the pipeline is composed by 5 main parts, for a total of 7 scripts, you can find them in the 'code' folder:

PART 1- BUILDING THE SUBTYPING MODEL OF AUTISM ADAPTIVE BEHAVIOR:
before srunning these script please fill the `utils.py` file tp set the working directories. Create an utils file including variables as reported in `example_utils.py` (you can find it in code/strat).Or modify the existing utils.py with you specific working directories/folders

**00A_NDAdata_wrangling.ipynb**. It runs on python and it cleans the raw data from NDA database. It split the clean dataset into train and test set.
**00B_edition preprocssing.Rmd**. It runs on R and applys a correction on the raw data removing variance given by using different VABS versions.
**01A_reval_clustering.ipynb**. It runs on python and produce the unsupervided stratification of NDAR VABS database using REVAL library.
**01B_plotREVALresults.Rmd**. It runs on R and produce plots and general visualization for REVAL outputs. 
**01C_plot_NDAR_longitudinal_VABS_MELS.Rmd**. It runs on R and it produces visualization and analysis for longitudinal trajecories of VABS subtypes on VABS and MSEL scores across the  (present in the 'code' folder as well). 

PART 2 - BUILD THE CLASSIFIER TO APPLY THE STRATIFICATION MODEL ON NOVEL DATASETS:
**02_UCSD_classification.ipynb** It runs on python. It applies the VABS clusters discovered in NDAR on the a new dataset: UCSD. this is a longitudinal dataset -> the clusters model is applied on the 1st time point available (only if the the 1st time point is before 72 months). In this script you can also find some code to execute the classification in any new database whose subject has been tested before 72 months (that must have the VABS domain scores for the subscales: communication, dailylivingskills, socialization)

PART 3 - APPLY THE SUBTYPING MODEL TO THE UCSD DATASET:
**03_UCSD_logitudinal_posthoc_analyis.Rmd** It runs on R. and produce the longitudinal analysis for the UCSD longitudinal datasets already clustered. Longitudinal analysis are provided for both VABS and MSEL

PART 4 -  COMPARE REVAL STRATIFICATION MODEL WITH A NORMATIVE MODEL
**04_model_comparison.Rmd** It runs on R. and produce a second possible subtyping model by applying norms from VABS manual, and comparare this model to REVAL stratification ones.

PART 5 - CREATE THE HYBRID MODEL THAT COMBINES THE LABELS FROM REVAL AND THOSE FROM VABS NORMS MODEL
**05_hybridmodel.Rmd.** It runs on R. and produce labels, visualizations and anlysis for the hybrid model as well as statistics for comparing it to REVAL and VABSnorm models.

PART 6 - OLDER SUBJECT STRATIFICATION
**06A_NDAOLD_data_wrangling.ipynb**. It runs on python and split the older dataset into train and test set.
**06B_OLD_edition_proprocessing.Rmd**. It runs on R and applys a correction to the old dataset for removing varinace given by using different VABS format.
**07A_OLD_runreval.ipynb**  this script runs on python and prosuce older subjects stratification model applying reval 
**07B_OLD_plotREVALresults.Rmd** this script run in R. it prodices plots and general visualization for REVAL outputs on older cohort
**08_young_old.Rmd** this script run in R. it prodices analysis and visualization for matching young and old subjects stratification.


In the 'code/strat' folder there are other minor scripts used to produced plots and figures and function recalled in the 3 main scripts.
In the'code/R' folder you can find minor function written in R that are recalled in the main scripts to produce plots of recurrent analysis.
