This project contains the Python code needed my dissertation: "Graph Neural Networks in Molecular Property prediction"

The code for section 4.2 is found in the NonGNN folder - run the non_gnn_clintox.py and non_gnn_esol.py files.

The code for section 4.3 is found in the BasicGNN folder - run the basics_clintox.py and basics_esol.py files.

The code for section 4.4 is found in the KnoMol folder - pick the appropriate folder corresponding to the type of KnoMol model you want to run, then run molnetdata.py followed by run_2.py. knomol_slurm_clintox_2.sh and knomol_slurm_esol_2.sh show some appropriate arguments to pass. The readme file within each subfolder contains instructions from the original KnoMol project authors on setting up an appropriate python environment.
The code for section 4.5 is found in the HimGNN folder - pick the appropriate folder corresponding to the type of HimGNN model you want to run, then run hyperparam_search_v4.py. himgnn_slurm_clintox_v4.sh and himgnn_slurm_esol_v4.sh show some appropriate arguments to pass. The readme file within each subfolder contains instructions from the original HimGNN project authors on setting up an appropriate python environment.
The code for chapter 5 is found in the ModelComparison folder - run the analyse_model.py file. 

The experiment_results.xlsx file shows the key results obtained from running these models. To inspect the line-by-line diagnostic output without having to run the model, find the appropriate slurm-xxxxx.out file, where xxxxx is found in the Slurm ID column. 
These output files can be found at https://drive.google.com/drive/folders/1KQxgbm3Q_n_LJHVa5vvUYEPOU-2bgoSt
