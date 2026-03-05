********************************************************************************
********************************************************************************
*** DISCERN 2.0 / Arora, A., Belenzon, B., Cioaca, L., Sheer, L., & Shvadron, D.
*** Updated: August 2024
********************************************************************************
********************************************************************************

* The main output files include:
	* Firm-year panel: discern_firm_panel_1980_2021
	* Granted patent dataset: discern_pat_grant_1980_2021
	* Pre-grant patent application dataset: discern_pat_app_1980_2021
	* Scientific publication dataset: discern_pub_1980_2021
	* List of ultimate owner names: discern_uo_names
	* List of subsidiary names: discern_uo_names
	* Crosswalk between the firm identifiers in DISCERN 2.0 and Compustat: permno_gvkey
	* List of representative firm names (standardized): name_permno_adj

* Researchers can use these files directly; we are providing the code files below as examples for how to pre-process the Compustat data and calculate accurate stocks of patents, patent applications, and publications while taking into consideration the dynamic ownership of subsidiaries and firms over time

********************************************************************************

* This is the main do file. It connects to all the other do files in the "code_examples" folder
* Users should read Section A of the "Data Appendix DISCERN DEC 2020" file available in Version 1.2.1 before using the data
* Due to intellectual property (IP) restrictions, we cannot redistribute all the input files; in such cases, we provide sample code for reference purposes

* Install required packages (if not already installed)
* ssc install gtools

* Set the Stata version
version 18
clear all
set more off

* Change the working directory
cd "/*enter your directory path here*/"

* Step 1: Preprocess the Compustat financial data
********************************************************************************
* Based on Compustat records obtained through WRDS in May 2022
* Users should obtain the Compustat North America > Fundamentals Annual data file before running "compustat_do.do" 

do "./code_examples/2_compustat.do"

* Step 2: Compile the granted patent data
********************************************************************************
* OUTPUTS: stock and stock variables considering the dynamic ownership of patents: "data/pat_grant_stock_permno_adj.dta" and "data/pat_grant_per_fyear_permno_adj.dta"

do "./code_examples/3_pat_grant.do"

* Step 3: Compile the pre-grant patent application data
********************************************************************************
* OUTPUTS: stock and flow variables considering the dynamic ownership of patent applications: "data/pat_app_stock_permno_adj.dta" and  "data/pat_app_per_fyear_permno_adj.dta"

do "./code_examples/4_pat_app.do"

* Step 4: Compile the publication data
********************************************************************************
* OUTPUTS: stock and flow variables considering the dynamic ownership of publications: "pub_stock_permno_adj.dta" and "data/pub_per_fyear_permno_adj.dta"

do "./code_examples/5_pub.do"

* Step 5: Compile the firm panel
********************************************************************************
* OUTPUT: firm panel: "./output_files/discern_firm_panel_1980_2021.dta"
* Note that "permno_adj" is the UO firm unique ID

do "./code_examples/6_firm_panel.do"

* EXTRA PROGRAM:
********************************************************************************
* The file “code_examples/name_std.do” provides sample code for standardizing the name lists

