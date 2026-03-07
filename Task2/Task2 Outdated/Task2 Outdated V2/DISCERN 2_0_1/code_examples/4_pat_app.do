********************************************************************************
********************************************************************************
*** DISCERN 2.0 / Arora, A., Belenzon, B., Cioaca, L., Sheer, L., & Shvadron, D.
*** Updated: August 2024
********************************************************************************
********************************************************************************

* Compile the pre-grant patent applications data: flow and stock variables considering the dynamic ownership of patents
* Data: U.S. patent applications from USPTO over 2000-2021 (post AIPA)
* INPUT FILE: Match file of UO and subsidiary patent applications, which links id_name to patents: "discern_pat_app_1980_2021.dta"
* OUTPUT FILES: "pat_app_stock_permno_adj.dta" and "pat_app_per_fyear_permno_adj.dta"

* Compute the stock of patent applications (similar to NBER2006 code)
********************************************************************************

use "./output_files/stata_files/discern_pat_app_1980_2021.dta", clear
drop permno_adj
gduplicates drop

* Count the number of assignees for each patent application
tostring id_name, gen(id_name_str)
gen id_name1 = sample+id_name_str
bysort application_id: egen nass = count(id_name1)

* Calculate the fractional patent application ownership
gen npat_app = 1 / nass

* Calculate the number of patent applications for each id_name-fyear
bysort id_name1 fyear: egen npat_app_id_name_fyear = sum(npat_app)

save "./pat_app_database", replace


use "./pat_app_database", clear
gduplicates drop id_name1 fyear, force

keep id_name1 fyear npat_app_id_name_fyear

* Calculate the name level patent application count
fillin id_name1 fyear
replace npat_app_id_name_fyear=0 if npat_app_id_name_fyear==.

gen id_name=substr(id_name1, 2,.)
gen sample=substr(id_name1, 1,1)
destring id_name, replace

* Dynamic match of assignee to UO firms
merge m:1 id_name sample using "./dyn_match_all.dta"
keep if _m==3
drop _m

* Identify the appropriate permno_adj to assign the patent applications to in each year
gen permno_adj=.
forvalue i=0/5 {
	replace permno_adj = permno_adj`i' if permno_adj`i'~=. & fyear >= fyear`i' & fyear <= nyear`i'
}

* Patent applications enter our sample once the related UO firm is publicly traded and not before
drop if fyear0 !=. & fyear < fyear0
drop if fyear0 ==. & fyear < fyear1

* Generate the patent application stock - 0.15% growth per year plus 15% depreciation
gen sum_npat_app_id_name = npat_app_id_name_fyear/0.15
bysort id_name1 (fyear): replace sum_npat_app_id_name=0.85*sum_npat_app_id_name[_n-1]+npat_app_id_name_fyear if sum_npat_app_id_name[_n-1]!=.

keep fyear id_name1 id_name npat_app_id_name_fyear _fillin sum_npat_app_id_name permno_adj sample
save "./id_name_fyear_npat_app", replace

* Sum over multiple names to get patent applications for each permno_adj 
use "./id_name_fyear_npat_app", clear
drop if permno_adj==.

gsort id_name1 fyear
gcollapse (sum) sum_npat_app_id_name, by(permno_adj fyear)
ren sum_npat_app_id_name pat_app_stock
label var pat_app_stock "Cumulative patent application stock"
label var permno_adj "UO firm unique ID"

* Drop in non-Compustat acquirers labeled as "99999"
drop if permno_adj==99999

save "./pat_app_stock_permno_adj.dta", replace

* Compute the flow of patent applications
********************************************************************************

use "./pat_app_database", clear
gduplicates drop id_name1 fyear, force
keep id_name1 id_name fyear npat_app_id_name_fyear sample

merge m:1 id_name sample using "./dyn_match_all.dta"
keep if _m==3
drop _m

* Identify the appropriate permno_adj to assign the patent applications to in each year
gen permno_adj=.
forvalue i=0/5 {
replace permno_adj = permno_adj`i' if permno_adj`i'~=. & fyear >= fyear`i' & fyear <= nyear`i'
}
drop if permno_adj == .
keep permno_adj fyear npat_app_id_name_fyear

* Sum over multiple id_names to get patent applications for each permno_adj-fyear
gsort permno_adj fyear
gcollapse (sum) npat_app_id_name_fyear, by(permno_adj fyear)

fillin permno_adj fyear
replace npat_app_id_name_fyear = 0 if npat_app_id_name_fyear == .

ren npat_app_id_name_fyear pat_app_fyear
drop _fillin

* Note that the file is balanced by firm-year (2000-2021) i.e., it contains years when the firm is not traded. These will drop when merged into the firm panel file.

label var pat_app_fyear "Flow of patent applications per permno_adj-year"
label var permno_adj "UO firm unique ID"
save "./pat_app_per_fyear_permno_adj", replace