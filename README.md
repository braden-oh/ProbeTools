# ProbeTools
### Braden Oh
This is a suite of tools that I wrote to help me post-process probe data at the University of Michigan's Plasmadynamics and Electric Propulsion Laboratory.  The tools use Bayesian inference to extract data with uncertainty from quartz crystal microbalance (QCM) data, Langmuir probe I-V traces, and retarding potential analyzer (RPA) sweeps.  All are works in progress and were originally written for the beam catcher campaign, so are specialized for those files.  I'll generalize as I have time to do so.  As of 08/26/2025, the QCM code is the only generalized code.

## QCM Code
### Precursory Information
* The scripts assume that there is exactly one header line of the form "XTC/3 Data Log 7/9/2025 3:35:18 PM."  This is the default header created by the XTC reader.  This requires that you do not stop and then restart recording in XTC without moving to a new filename.  If you do do this, then ensure you delete any extraneous headers.
* The parser requires that the output from XTC includes the following items: *Timestamp, Rate, Thickness, State Time, Raw Frequency*
* Requires Python libraries *pymc, pandas, and numpy*

* When you run the parser, the extracted data will be saved by default to a file called ```QCM Database.csv``` in the target directory.
* The parser will fit a line to each dataset using MCMC to learn (with uncertainty) the deposition rate
* The parser will also fit a line to each dataset using RANSAC over the entire dataset.  This serves as a sanity check.
* The parser will save images of the Bayesian and RANSAC line fits in a folder called ```Figures``` in the target directory.

### Usage
1. Open a terminal window and navigate to the directory containing QCM.py (this would be the ProbeTools directory).
2. Call the script from command line using one of the following: 
    * ```python QCM.py /data_folder --interact``` to process only new data files in interactive mode
    * ```python QCM.py /data_folder --fitall --interact``` to process all data files in interactive mode
    * ```python QCM.py /data_folder --fitall``` to process all data files using the entire dataset
    * ```python QCM.py /data_folder``` to process only new data files using the entire dataset
3. Find the extracted data in the table ```/data_folder/QCM Database.csv```
4. Inspect the Bayesian line fits by inspecting the graphs in ```/data_folder/Figures/Bayesian```
5. Inspect the RANSAC line fits by inspecting the graphs in ```/data_folder/Figures/RANSAC```

The above commands will execute the parser over all QCM files in the directory specified by ```data_folder```.  The flag ```--fitall``` tells the command to process every file in the directory.  If you've added new files and don't want to repeat processing earlier files, omit the flag.  The flag ```--interact``` launches the parser in an interactive mode that will allow you to select what portions of the data you want to fit to.  If you want to fit to the entire dataset without user input, omit the flag.

### Specific Commands 
If you're interested in writing your own parser or running from Python scripts, these two commands may be of interest to you:
* The ```load_QCM(path, bins=5)``` command loads data from a single file.  The command averages measurements by binning the data.  It defaults to 5 second bin widths.
* The ```process_directory(path)``` command will parse an entire directory by searching for filenames containing the string 'QCM' and ending in '.txt'.
