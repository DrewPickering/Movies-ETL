# Movies-ETL
Extract Wikipedia and Kaggle movie data to be cleaned and loaded into database

# Challenge Amazing Prime

Amazing Prime and online video service provider has sponsored a hackathon to extract, transform, and load data into a database to be analyzed for the purpose of finding low cost popular movies to purchase for content on their service. 

Data has been extracted from Wikipedia and Kaggle, cleaned and organized and loaded into an SQL database.

The updates for this extract, transform, and load have been automated, so that they can be performed simply and easily.  During the automation process, additional error proofing of potential failures have been inserted in the form of try and except blocks of code to factor out potential faulty assumptions about the consistency of data sources. 

Assumptions:  

1.	Wiki Budget remove value between $ and a hyphen will capture all variations of inputs into the field; try/except block added to skip movie if an error occurs.
2.	Wiki Budget forms and apply parse_dollar function will not result in an error; try/except block added to skip movie if an error occurs.
3.	Wiki release_date string conversion and forms application will not result in an error: try/except block added to skip movie if an error occurs.
4.	Specific merge error on release_date will not generate an error; try/except block added to skip movie if an error occurs.
5.	Dropping unneeded Wiki columns will not produce error; try/except block added to skip column if Wiki column with column header does not exist.
6.	Kaggle runtime missing data fill with Wiki running_time; try/except block added to set runtime to 0 if Wiki running_time does not exist.
7.	Kaggle budget_kaggle missing data fill with Wiki budget_wiki; try/except block added to set budget_kaggle to 0 if budget_wiki does not exist.
8.	Kaggle revenue missing data fill with Wiki box_office; try/except block added to set revenue to 0 if box_office does not exist.
