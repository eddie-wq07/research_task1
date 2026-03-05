********************************************************************************
********************************************************************************
*** DISCERN 2.0 / Arora, A., Belenzon, B., Cioaca, L., Sheer, L., & Shvadron, D.
*** Updated: August 2024
********************************************************************************
********************************************************************************

* Preprocess the Compustat financial data 
* We obtained the Compustat North America > Fundamentals Annual data file through WRDS in May 2022; file name: "h4trvvtuu2v7km4x.dta"
* We cannot provide the Compustat data due to IP restrictions
* Users should obtain the Compustat data file and place it in the working directory before running the code below

* Clean the Compustat file
use "./h4trvvtuu2v7km4x.dta", clear

* Drop financial services firms
keep if indfmt == "INDL"

* Drop fiscal years outside panel period
drop if fyear == .
drop if fyear > 2021
drop if fyear < 1980

* Drop Canadian stock exchanges
drop if exchg == 7 | exchg == 8 | exchg == 9 | exchg == 10

* When we created the DISCERN dataset, we dropped firms not headquartered in the U.S. (loc = Current ISO Country Code - Headquarters)
*keep if loc == "USA"

* When we created the DISCERN dataset, we dropped firms that don't perform R&D
/*
bysort gvkey (fyear): egen mxrd = mean(xrd)
drop if mxrd == . | mxrd <= 0
drop mxrd
*/

* Drop duplicates in terms of gvkey-fyear
gsort gvkey fyear
drop if gvkey == gvkey[_n-1] & fyear == fyear[_n-1]

* When we created the DISCERN dataset, we kept only years with traded shares
/*
gsort gvkey fyear
gen miny = (fyear * (cshtr_f != .))
replace miny = . if miny == 0
egen miny1 = min(miny), by(gvkey)
drop if fyear <= miny1 - 1
drop miny miny1

gsort gvkey fyear
egen maxy = max(fyear*(cshtr_f != .)), by(gvkey)
drop if fyear >= maxy+1
drop maxy

* Drop years after the last year with sales data
gsort gvkey fyear
egen maxy = max(fyear*(sale != .)), by(gvkey)
drop if fyear >= maxy + 1
drop maxy

* Drop years before the first year with sales data
gsort gvkey fyear
gen miny = (fyear*(sale != .))
replace miny = . if miny == 0
egen miny1 = min(miny), by(gvkey)
drop if fyear <= miny1 - 1
drop miny miny1
*/

save "./compustat_sample_1980_2021.dta", replace