********************************************************************************
********************************************************************************
*** DISCERN 2.0 / Arora, A., Belenzon, B., Cioaca, L., Sheer, L., & Shvadron, D.
*** Updated: August 2024
********************************************************************************
********************************************************************************

* Compile the granted patent data: flow and stock variables considering the dynamic ownership of patents
* Data: U.S. granted patents from USPTO over 1980-2021
* INPUT FILE: Match file of UO and subsidiary patents, which links id_name to patents: "discern_pat_grant_1980_2021.dta"
* OUTPUT FILES: "pat_grant_stock_permno_adj.dta" and "pat_grant_per_fyear_permno_adj.dta"

* Combine the UO and subsidiary names and their dynamic ownership
********************************************************************************

use "./output_files/stata_files/discern_uo_names.dta", clear
append using "./output_files/stata_files/discern_sub_names.dta"
keep sample id_name fyear* nyear* permno_adj*
label var sample "U=UO+Labs+traded-Subs//S=SEC Exhibit 21"
save "./dyn_match_all.dta", replace

* Sample code to identify the relevant firm (permno_adj) at the patent grant year for each patent whose assignee was name-matched to a DISCERN ultimate owner or subsidiary name
********************************************************************************
* Note: this step has already been completed for the patent file "discern_pat_grant_1980_2021.dta"; the code below serves as an example only

/*
use patent_id-sample using "./output_files/stata_files/discern_pat_grant_1980_2021.dta", clear

merge m:1 id_name sample using "./dyn_match_all.dta"
keep if _m == 3
drop _m

* Identify the appropriate permno_adj to assign the patent to at grant year
gen permno_adj=.
forvalue i=0/5 {
	replace permno_adj = permno_adj`i' if permno_adj`i'~=. & fyear >= fyear`i' & fyear <= nyear`i'
}

* Granted patents enter our sample once the related UO firm is publicly traded and not before. 
* Furthermore, we do not account for patents in gap years when the matched UO or subsidiary is not publicly traded.
drop if permno_adj == .

keep patent_id patent_date assignee_name fyear name_std id_name sample permno_adj

label var permno_adj "UO firm unique ID"

save "./output_files/stata_files/discern_pat_grant_1980_2021.dta", replace
*/

* Compute the stock of granted patents (similar to NBER2006 code)
********************************************************************************

use "./output_files/stata_files/discern_pat_grant_1980_2021.dta", clear
drop permno_adj
gduplicates drop

* Count the number of assignees for each patent
tostring id_name, gen(id_name_str)
gen id_name1 = sample+id_name_str
bysort patent_id: egen nass = count(id_name1)

* Calculate the fractional patent ownership
gen npat = 1 / nass

* Calculate the number of patents for each id_name-fyear
bysort id_name1 fyear: egen npat_id_name_fyear = sum(npat)

save "./pat_grant_database", replace


use "./pat_grant_database", clear
gduplicates drop id_name1 fyear, force

keep id_name1 fyear npat_id_name_fyear

* Calculate the name level patent count
fillin id_name1 fyear
replace npat_id_name_fyear=0 if npat_id_name_fyear==.

gen id_name=substr(id_name1, 2,.)
gen sample=substr(id_name1, 1,1)
destring id_name, replace

* Dynamic match of assignee to UO firms
merge m:1 id_name sample using "./dyn_match_all.dta"
keep if _m==3
drop _m

* Identify the appropriate permno_adj to assign the patents to in each year
gen permno_adj=.
forvalue i=0/5 {
	replace permno_adj = permno_adj`i' if permno_adj`i'~=. & fyear >= fyear`i' & fyear <= nyear`i'
}

* Granted patents enter our sample once the related UO firm is publicly traded and not before
drop if fyear0 !=. & fyear < fyear0
drop if fyear0 ==. & fyear < fyear1

* Generate the patent stock - 0.15% growth per year plus 15% depreciation
gen sum_npat_id_name=npat_id_name_fyear/0.15
bysort id_name1 (fyear): replace sum_npat_id_name=0.85*sum_npat_id_name[_n-1]+npat_id_name_fyear if sum_npat_id_name[_n-1]!=.

keep fyear id_name1 id_name npat_id_name_fyear _fillin sum_npat_id_name permno_adj sample
save "./id_name_fyear_npat", replace

* Sum over multiple names to get patents for each permno_adj 
use "./id_name_fyear_npat", clear
drop if permno_adj==.

gsort id_name1 fyear
gcollapse (sum) sum_npat_id_name, by(permno_adj fyear)
ren sum_npat_id_name pat_grant_stock
label var pat_grant_stock "Cumulative patent stock"
label var permno_adj "UO firm unique ID"

* Drop in non-Compustat acquirers labeled as "99999"
drop if permno_adj==99999

save "./pat_grant_stock_permno_adj.dta", replace

* Identify patenting firms (i.e, firms with positive patent stock)
********************************************************************************

use "./pat_grant_stock_permno_adj.dta", clear
drop if pat_grant_stock == 0
keep permno_adj
gduplicates drop
save "./patenting_firms.dta" , replace

* Compute the flow of granted patents
********************************************************************************

use "./pat_grant_database", clear
gduplicates drop id_name1 fyear, force
keep id_name1 id_name fyear npat_id_name_fyear sample 

merge m:1 id_name sample using "./dyn_match_all.dta"
keep if _m==3
drop _m

* Identify the appropriate permno_adj to assign the patents to in each year
gen permno_adj=.
forvalue i=0/5 {
replace permno_adj = permno_adj`i' if permno_adj`i'~=. & fyear >= fyear`i' & fyear <= nyear`i'
}
drop if permno_adj == .
keep permno_adj fyear npat_id_name_fyear

* Sum over multiple id_names to get patents for each permno_adj-fyear
gsort permno_adj fyear
gcollapse (sum) npat_id_name_fyear, by(permno_adj fyear)

fillin permno_adj fyear
replace npat_id_name_fyear=0 if npat_id_name_fyear==.

ren npat_id_name_fyear pat_grant_fyear
drop _fillin

* Note that the file is balanced by firm-year (1980-2021) i.e., it contains years when the firm is not traded. These will drop when merged into the firm panel file.

label var pat_grant_fyear "Flow of granted patents per permno_adj-year"
label var permno_adj "UO firm unique ID"
save "./pat_grant_per_fyear_permno_adj", replace