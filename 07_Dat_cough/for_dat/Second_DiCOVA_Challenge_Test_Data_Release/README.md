#############################################################################################
#                                                                                           # 
#                             The Second DiCOVA 2021 Challenge                              #
#                            Diagnosing COVID-19 Using Acoustics                            #
#                                   Evaluation Data Release			            #
#                               http://dicovachallenge.github.io/                           #
#                                                                                           #  
#############################################################################################

1. About
While the list of symptoms of COVID-19 infection is regularly updated, it is established that
in symptomatic cases COVID-19 seriously impairs normal functioning of the respiratory system.
Does this alter the acoustic characteristics of breathe, cough, and speech sounds produced
through the respiratory system? This is an open question waiting for scientific insights.
The DiCOVA Special Session/Challenge is designed to find scientific and engineering insights
to the question by enabling participants to analyze an acoustic dataset gathered from COVID-19
positive and non-COVID-19 individuals. Here we provide the blind dataset for use in Evaluation.

2. Directory structure
```
│   README.md  
│   LICENSE.md    
│   metadata.csv  
└───AUDIO
		| breathing
    		│   <file_name>.flac
		| cough
    		│   <file_name>.flac
		| speech
    		│   <file_name>.flac
```

3. Directory contents
AUDIO              	: Contains 471 audio files for each audio category, stored in .FLAC format.
LICENSE.md         	: Contains the Distribution License
metadata.csv  		: Contains the subject information for each audio file.
	
4. Audio file description
Each audio file, in each audio category, corresponds to a unique subject. Some metadata of the subjects is provided in
metadata.csv. The audio file details are as follows: 
	- Sampling Rate : 44.1 kHz
	- Channels      : 1
	- Precision     : 16-bit	
	- Format        : .FLAC

5. metadata.csv header description
BREATHING_ID: <FILE_ID>
COUGH_ID: <FILE_ID>
SPEECH_ID: <FILE_ID>
FUSION_ID: <FUSION_ID>
COVID_STATUS: To be inferred by participant.
GENDER      : Male (m) / Female (f)

6. Instructions
- Each row in metadata.csv corresponds to a unique subject.
- The BREATHING_ID, COUGH_ID, SPEECH_ID, FUSION_ID are to be used while uploading scores to the corresponding tracks in the leaderboard.
- You can use the DiCOVA Train-Val Data in any manner to train your models for making
inference on the "COVID_STATUS" of the Evaluation Data.
- Please adhere to the Terms and Conditions document signed by your team.

7. Contact Us
Please reach out to dicovachallenge@gmail.com for any queries.

8. Organizers
    Team DiCOVA
    - Sriram Ganapathy | Assistant Professor, IISc, Bangalore
    - Prasanta Kumar Ghosh | Associate Professor, IISc, Bangalore
    - Neeraj Kumar Sharma | CV Raman Postdoctoral Researcher, IISc, Bangalore
    - Srikanth Raj Chetupalli | Postdoctoral Researcher, IISc, Bangalore
    - Debarpan Bhattacharya | MTech Student, IISc, Bangalore
    - Debottam Dutta | Research Associate, IISc, Bangalore
    - Pravin Mote | Senior Research Fellow, IISc, Bangalore


#############################################################################################
