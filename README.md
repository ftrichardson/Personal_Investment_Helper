# Personal_Investment_Helper

How the run the program: Please execute Project.py to run the program.


Main files:

REAMEME.md : This file

Project.py :  Main program 

factor.py : Part of the main code that is responsible for the multi factor mode building. Separated for debugging purpose.

wb.jpg :  A image used in the main program

crawler : a folder.
	
    	web_crawling_final.py: Scrape the needed information of 30 Dow Jones stocks by using Selenium. Import username and password to log in and filter data accordingly. Create a csv named as {company_ticker}.csv, saved under the folder named data.
    
    	company_info.csv: Use Selenium to extract company information and save as a csv file.

    	data: web_crawling_final.py creates 30 csv files here

	data_backup: In order to save time and avoid potential error, we make this directory to include 30 csv files already extracted by Selenium. Our factor.py also calls from this path. 

Function_description.pdf: Brief introduction of the main functions the program has, as well as the requirement of inputs.

RPS.py: A measure to select the best stocks (didn't put in the Project.py)
