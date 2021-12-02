
I/ Folder Hierachy:

01_asc
  |--requirement.txt   : file contains package version
  |--README.txt        : file README
  |
  |--01_input          : folder contains video input (<file>.mp4). There is one file.mp4 here for testing. You may put new files at here for testing
  |--02_sys            : folder contains systems


II/ Steps to run experiments

01/ Install conda environment
+ Download conda by command line:  
    wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh

+ Install conda by the command line: 
    bash Anaconda3-2019.03-Linux-x86_64.sh

02/ Install packages/libraries by the command line: 
    conda create --name test_03 --channel conda-forge --channel pytorch --file requirement.txt

03/ Activate the environment by the command line: 
    conda activate test_03

04/ Install torchlibrosa by pip
    pip install torchlibrosa

04/ Go into folder '02_sys', then run the command line 'run.sh' to generate the folder '03_output'
 
01_asc
  |--requirement.txt   : file contains package version
  |--README.txt        : file README
  |
  |--01_input          : folder contains video input (file.mp4)            
  |--02_sys            : folder contains systems
  |
  |--03_output         : folder contains report files --> This folder is generated after step 04 with 'run.sh'

05/ Check the report for:
     + Noise Environment Classification:  './03_output/04_asc_report/<file>.csv'
     + Sound Event Detection:             './03_output/05_aed_report/<file>.py
     + Description of Sound Context:      './03_output/06_aud_des/<file>.txt'




