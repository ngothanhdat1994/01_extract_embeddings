#############################################################################################
#                                                                                           # 
#                                The Second DiCOVA Challenge                                #
#                            Diagnosing COVID-19 Using Acoustics                            #
#                                  Development Data Release                                 #
#                             https://dicovachallenge.github.io/                            #
#                                                                                           #  
#############################################################################################

1. About
While the list of symptoms of COVID-19 infection is regularly updated, it is established that
in symptomatic cases COVID-19 seriously impairs normal functioning of the respiratory system.
Does this alter the acoustic characteristics of breathe, cough, and speech sounds produced
through the respiratory system? This is an open question waiting for scientific insights. The
Second DiCOVA Special Challenge is designed to find scientific and engineering insights to
the question by enabling participants to analyze an acoustic dataset gathered from individuals
with and without COVID-19. Here we provide the development dataset to be used in the challenge.

2. Directory structure
```
│   README.md  
│   LICENSE.md    
│   metadata.csv  
└───AUDIO
│   │
│   └─── <SOUND_CATEGORY>
│        │ 
│        └───<subject_ID>.flac
└───LISTS
    │   train_<fold_num>.csv
    │   val_<fold_num>.csv
```

2. Directory contents
    AUDIO       : Contains sound category-wise audio files
    LISTS       : Contains train and validation lists for 5 folds
    LICENSE.md  : Contains the License information
    metadata.csv: Contains the subject information associated with each audio file
	
3. Audio file description
Each audio file corresponds to a unique subject. Some metadata of the subjects is provided in
metadata.csv. The audio file details are as follows: 
	- Sampling Rate : 44.1 kHz
	- Channels      : 1
	- Precision     : 16-bit	
	- Format        : FLAC

4. metadata.csv header description
	SUB_ID         : <SUBJECT_ID>
	COVID_STATUS   : COVID+ve (p) / Non-COVID (n)
	Gender         : Male (m) / Female (f)

5. Instructions
-   Adhere to the train-val folds while reporting the validation performance.
-   Adhere to the Terms and Conditions document signed by your team.
-   This dataset is released for use in the Second DiCOVA Challenge only.

6. Citation
Please refer to the below publication to know more about the data collection methodology used.
Also, cite this publication on using this dataset anywhere.
-   Sharma, N., Krishnan, P., Kumar, R., Ramoji, S., Chetupalli, S.R., R., N., Ghosh, P.K.,
    Ganapathy, S. (2020) "Coswara — A Database of Breathing, Cough, and Voice Sounds for
    COVID-19 Diagnosis", Proc. Interspeech 2020, 4811-4815, DOI: 10.21437/Interspeech.2020-2768

7. Contact Us
Please reach out to dicova.challenge@gmail.com (or any of the organizers below) for any queries.

8. Organizers
    Team DiCOVA
    - Sriram Ganapathy | Assistant Professor, IISc, Bangalore*
    - Prasanta Kumar Ghosh | Associate Professor, IISc, Bangalore
    - Neeraj Kumar Sharma | CV Raman Postdoctoral Researcher, IISc, Bangalore
    - Srikanth Raj Chetupalli | Postdoctoral Researcher, IISc, Bangalore
    - Debarpan Bhattacharya | MTech Scholar, IISc, Bangalore
    - Debottam Dutta | Research Associate, IISc, Bangalore
* Indian Institute of Science, Bangalore-560012, India

#############################################################################################
