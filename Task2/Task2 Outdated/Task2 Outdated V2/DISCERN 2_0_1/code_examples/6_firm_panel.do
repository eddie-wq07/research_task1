********************************************************************************
********************************************************************************
*** DISCERN 2.0 / Arora, A., Belenzon, B., Cioaca, L., Sheer, L., & Shvadron, D.
*** Updated: August 2024
********************************************************************************
********************************************************************************

* Compile the firm panel and merge in granted patents, pre-grant patent applications, and publications
* OUTPUT: panel file:"./output_files/discern_firm_panel_1980_2021"
* Note that "permno_adj" is the UO firm unique ID

use "./output_files/stata_files/permno_gvkey.dta", clear
expand 42 // replaces each observation in the dataset with 42 copies of the observation
bysort permno_adj gvkey: gen fyear = 1980 + (_n-1)

* When more than one GVKEY is related to a permno-adj, we keep only one record per firm-year (that has Compustat data)
gen t = (fyear >= fyear1_adjust & fyear <= fyearn_adjust)
gsort permno_adj fyear -t 
gduplicates drop permno_adj fyear, force
keep if fyear >= min_y_permno & fyear <= max_y_permno

* Identify gap years
replace gvkey = "" if t==0 // 52 real changes made, 52 to missing

* Merge-in the financial data - users shoud download Compustat data and run "2_compustat.do" in advance
********************************************************************************

merge m:1 gvkey fyear using "./compustat_sample_1980_2021.dta"
drop if _m==2

* Identify additional Compustat gap years
replace gvkey = "" if _m==1
replace t = 0 if _m==1
drop _m
gsort permno_adj fyear

* Drop too large of a gap w/o Compustat data (10 years of gap and then 3 more years active - we drop the 13 observations)
/* how to check gap:
gen t2=t==0
by permno_adj: egen gap=total(t2)
*/
drop if permno_adj==64493 & fyear > 1987


* Merge-in granted patent variables to main panel
********************************************************************************

merge 1:1 fyear permno_adj using "./pat_grant_stock_permno_adj.dta" 
drop if _m==2
drop _m

merge 1:1 fyear permno_adj using "./pat_grant_per_fyear_permno_adj"
drop if _m==2
drop _m

foreach var of varlist pat_grant_stock pat_grant_fyear {
	replace `var'=0 if  `var'==.
}

* Merge-in patent application variables to main panel
********************************************************************************

merge 1:1 fyear permno_adj using "./pat_app_stock_permno_adj.dta" 
drop if _m==2
drop _m

merge 1:1 fyear permno_adj using "./pat_app_per_fyear_permno_adj"
drop if _m==2
drop _m

foreach var of varlist pat_app_stock pat_app_fyear {
	replace `var'=0 if  `var'==.
}

* Merge-in publication variables to main panel
********************************************************************************

merge 1:1 fyear permno_adj using "./pub_stock_permno_adj.dta" 
drop if _m==2
drop _m

merge 1:1 fyear permno_adj using "./pub_per_fyear_permno_adj.dta"
drop if _m==2
drop _m

foreach var of varlist pub_stock pub_fyear {
	replace `var'=0 if  `var'==.
}

drop fyear1_adjust fyearn_adjust min_y_permno max_y_permno
gsort permno_adj fyear

* Keep only variables that can be redistributed 
keep permno_adj gvkey fyear pat_grant_fyear pat_app_fyear pub_fyear
order permno_adj gvkey fyear pat_grant_fyear pat_app_fyear pub_fyear

* Bring in the most recent representative firm name (standardized)
merge m:1 permno_adj using "output_files/stata_files/name_permno_adj.dta"
drop if _m == 2
drop _m

label var fyear "Data Year - Fiscal"
label data "DISCERN 2.0 - Arora, Belenzon, Cioaca, Sheer, Shvadron - 2024"

compress
save "./output_files/stata_files/discern_firm_panel_1980_2021.dta", replace
export delimited "./output_files/csv_files/discern_firm_panel_1980_2021.csv", replace
