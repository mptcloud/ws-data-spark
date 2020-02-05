### EQ Works work sample 

Here, we outline the files and folder structure of the project.

#### Package(s) Installation

In order to install Python packages used within the various notebooks, an environment.yml file has been provided for your 
reference. You can create a conda environment using the following command:

```bash
conda env create -f environment.yml
```

The command will above the requisite environment with the name: ml_eq.

#### ws-data-spark Folder

##### Relevant sub-folders

1. `notebooks` -- Initial development work done on Jupyter and Databricks Spark notebooks
2. `output` -- Output files returned by the relevant scripts within the scripts folder. If the scripts are run, all the
output files will be saved in this folder 
3. `plots` -- Relevant figures and plots used to tackle the modeling part of the assignment
4. `scripts` -- Pyspark and Python scripts that run the analysis portion of the assignment
5. `modeling` -- Jupyter notebooks used to analyze the data following the pre-processing steps run via the scripts in `scripts`
6. `modeling_bonus.txt` -- A file explaining the rationale and underlying ideas behind some of the modeling approaches used

#### Notes

`scripts/python/python_analysis.py`

Once the requisite conda environment has been created, and the git repository cloned, then running the python file is as
simple as running the following commands in succession(which may change depending on where the repository has been 
stored):

```bash
conda activate ml_eq
python /home/<user-name>/ws-data-spark/scripts/python/python_analysis.py
```

Running the python scripts will populate the output folder and print the following output to the terminal (While the 
output below is an actual example run on my laptop, the numbers you see may change depending on the processor 
configuration):

```
Time elapsed for the iterative Pandas process: 170.0625 seconds
Time elapsed for the Pandas process using parallel map and reduce with 2 parallel processes: 2.8125 seconds
```

`scripts/pyspark/spark_analysis.py`

Assuming that the `$SPARK_HOME` environment variable points to your local Spark installation folder, then the pyspark job can 
be run from the project's root directory using the following command from the terminal:

```bash
$SPARK_HOME/bin/spark-submit \
--master local[*] \
/home/<user-name>/ws-data-spark/scripts/pyspark/spark_analysis.py
```

`modeling` folder

An attempt at answering the modeling questions can be found within the notebooks inside the modeling folder. As seen earlier 
in the analysis portion of the work sample, speeding up the process by using a map and reduce process or even PySpark is 
entirely possible for larger datasets. However, owing to the exploratory and investigative nature of the work in the 
modeling portion of the work sample, I will be using the notebooks within this folder as the basis of the discussion in 
`modeling_bonus.txt`.

## Contributor
Hari A Ravindran, Sr. Data Scientist, Canadian Tire
 