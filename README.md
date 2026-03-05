# DOW AAMD $\rightarrow$ CGMD Workflow

## Docker Usage and Setup

To set up the docker environment, on the command line first run `./build.sh` (if you get a permission denied error you may need to `chmod +x ./build.sh` first). This will take a bit of time, but only needs to be run once. You should rebuild only if the docker image changes. Then run `./interactive.sh` (again you may need to `chmod +x ./interactive.sh`), which will launch you into the docker container. 

## Code Usage

### Simulation

To run an AAMD simulation, on the command line within the docker container run the following:

`python src/run_aamd.py <system_config_file_name> <general_config_file_name>`

The system config file name will be used as the simulation identifier, and a trajectory file will be written out to the project folder as `system_config_file_name.dcd`. Note that you do not need to provide the extension or full path to these files, just the name. By default it will search within the expected `system_config_files` and `general_config_files` folders. If the files are not found there, it will also search in the main directory. 

### Visualizations

To generate plots of the quantities openMM logs for a given project, use the command `python src/plot_aamd.py <project name>`. This will create plots of temperature, density, volume, and energies, and write them out to a plot folder within the project directory. 

To visualize the trajectory file for a given project, in a jupyter notebook cell run `from src.aamd_utils import visualize_trajectory`, and then run `view = visualize_trajectory("<project name>")`, and then `view`. This will display an interactive movie of the trajectory.  

### Example

An example of running these commands for the mixed packed system, NVT simulation, and the default general config file (using openff forcefield) would be `python src/run_aamd.py mixed_nvt gen_default`. This creates a folder Project_Mixed with the trajectory dcd, topology pdb, and simulation log files. You could then plot the quantities from the log.txt file using `python src/plot_aamd.py Project_Mixed`. 

## Config Files

There are two configuration files required for each simulation run. The system config file defines parameters relevant to a given system and simulation. The general config file includes parameters related to how openMM performs its molecular dynamics simulations. The following sections describe the parameters contained in each config file. 

### System Config

* `project name`
    * **Meaning** : Name of the project corresponding to this system. Running a simulation will create a folder with the same name as the project. All results will be output to this folder.  
* `solvents`
    * **Meaning** : SMILES string(s), separated by spaces, corresponding to the solvent compound(s) in the system. 
    * **Options** :
        * O (Water)
        * CCCCCCCCCCCC (Dodecane)
        * OCCCCCCCC (1-octanol)
* `number of solvent molecules`
    * **Meaning** : Integer numbers, separated by spaces, corresponding to how many of each solvent compound(s) the system contains. 
* `density of solvents`
    * **Meaning** : Float numbers, separated by spaces, corresponding to the density of each solvent in $g/cm^3$
* `compounds of interest`
    * **Meaning** : SMILES string(s), separated by spaces, corresponding to the compound(s) of interest in the system. 
    * **Options** :
        * OCCOCCOCCOCCCCCCCCC (C9E3, triethylene glycol monononyl ether)
        * OCCOCCOCCCCCC (C6E2, tiethylene glycol monohexyl ether)
        * O1CCOCC1 (1,4-dioxane)
* `number of COI molecules`
    * **Meaning** : Integer numbers, separated by spaces, corresponding to the number of each compound of interest the system contains. 
* `density of COI`
    * **Meaning** : Float numbers, separated by spaces, corresponding to density of COI molecule(s) in $g/cm^3$. 
* `mixture type`
    * **Meaning** : Mixture type of the system, i.e. how a two-phase system is integrated. 
    * **Options** :
        * 0 : Fully mixed
        * 1 : Separated
        * 2 : COI dissolved into specified solvent
* `temperature`
    * **Meaning** : Temperature of the system, in Kelvin. 
* `pressure`
    * **Meaning** : Pressure of the system, in bars. Only used for NPT simulations. 
*  `simulation type`
    * **Meaning** : Statistical ensemble type of the simulation, defines which parameters are held constant.  
    * **Options** :
        * NVT : constant number, volume, temperature
        * NPT : constant number, pressure, temperature


### General Config

* `forcefield`
    * **Meaning** : Type of forecefield used for openMM molecular dynamics, defines the physics of how particles interact. 
    * **Options** : 
        * openff
        * amber
* `water model`
    * **Meaning** : Whether to explicitly use a water specific forcefield for a system containing water, and which model to use. 
    * **Options** : 
        * None
        * tip3
        * tip4
* `LJ interaction cutoff`
    * **Meaning** : Cutoff radius (in nm) past which Lennard Jones interaction potential is not calculated. 
    * **Default** : 1 nm
* `periodic box margin`
    * **Meaning** : Margin (percentage) past the initial packed box dimensions which will represent the periodic boundary cutoff. 
    * **Default** : 0
* `friction`
    * **Meaning** : Defines how strongly the system is coupled to its "heat bath", i.e. how tightly the temperature is regulated. 
    * **Default** : 1 picoseconds $^{-1}$
* `integration timestep`
    * **Meaning** : Time step (in femtoseconds) at which molecular dynamics forces are recomputed and the system state is advanced. 
    * **Default** : 2 femtoseconds
* `equilibration time`
    * **Meaning** : Time (in nanoseconds) that the system will spend equilibrating prior to the production run.
    * **Default** : 0.5 nanoseconds
* `production time`
    * **Meaning** : Time, in nanoseconds, that the system will run production for (only this period is used collect statistics for CG comparisons).
    * **Default** : 1 nanosecond
* `trajectory log frequency`
    * **Meaning** : How frequently (in nanoseconds) the system state is logged to the trajectory file. 
    * **Default** : 0.001 nanoseconds
* `pressure enforcing frequency`
    * **Meaning** : How frequently (in nanoseconds) the simulation will enforce the barostat pressure. Only relevant for NPT simulations. 
    * **Default** : 0.00005 nanoseconds