********************************************************************************
********************************************************************************
*** DISCERN 2.0 / Arora, A., Belenzon, B., Cioaca, L., Sheer, L., & Shvadron, D.
*** Updated: August 2024
********************************************************************************
********************************************************************************

* Compile the publication data: flow and stock variables considering the dynamic ownership of publications
* Data: Scientific publications (i.e., journal articles) from OpenAlex over 1980-2021
* INPUT FILE: Match file of UO and subsidiary publications, which links id_name to publications: "discern_pub_1980_2021.dta"
* OUTPUT FILES: "pub_stock_permno_adj.dta" and "pub_per_fyear_permno_adj.dta"

* Compute the stock of publications (similar to NBER2006 code)
********************************************************************************

use "./output_files/stata_files/discern_pub_1980_2021.dta", clear

* Note that each record pertains to an author-publication-name match
drop permno_adj author_id author_orcid author_name author_affil
gduplicates drop

* Count the number of matched authors for each publication
tostring id_name, gen(id_name_str)
gen id_name1 = sample+id_name_str
bysort openalex_id: egen naut = count(id_name1)

* Calculate the fractional publication authorship
gen npub = 1 / naut

* Calculate the number of publications for each id_name-fyear
bysort id_name1 fyear: egen npub_id_name_fyear = sum(npub)

save "./pub_database", replace


use "./pub_database", clear
gduplicates drop id_name1 fyear, force

keep id_name1 fyear npub_id_name_fyear

* Calculate the name level publication count
fillin id_name1 fyear
replace npub_id_name_fyear=0 if npub_id_name_fyear==.

gen id_name=substr(id_name1, 2,.)
gen sample=substr(id_name1, 1,1)
destring id_name, replace

* Dynamic match of affiliation to UO firms
merge m:1 id_name sample using "./dyn_match_all.dta"
keep if _m==3
drop _m

* Identify the appropriate permno_adj to assign the publications to in each year
gen permno_adj=.
forvalue i=0/5 {
	replace permno_adj = permno_adj`i' if permno_adj`i'~=. & fyear >= fyear`i' & fyear <= nyear`i'
}

* Publications enter our sample once the related UO firm is publicly traded and not before
drop if fyear0 !=. & fyear < fyear0
drop if fyear0 ==. & fyear < fyear1

* Generate the publication stock - 0.15% growth per year plus 15% depreciation
gen sum_npub_id_name = npub_id_name_fyear/0.15
bysort id_name1 (fyear): replace sum_npub_id_name=0.85*sum_npub_id_name[_n-1]+npub_id_name_fyear if sum_npub_id_name[_n-1]!=.

keep fyear id_name1 id_name npub_id_name_fyear _fillin sum_npub_id_name permno_adj sample
save "./id_name_fyear_npub.dta", replace

* Sum over multiple names to get publications for each permno_adj 
use "./id_name_fyear_npub.dta", clear
drop if permno_adj==.

gsort id_name1 fyear
gcollapse (sum) sum_npub_id_name, by(permno_adj fyear)
ren sum_npub_id_name pub_stock
label var pub_stock "Cumulative publication stock"
label var permno_adj "UO firm unique ID"

* Drop in non-Compustat acquirers labeled as "99999"
drop if permno_adj==99999

save "./pub_stock_permno_adj.dta", replace

* Compute the flow of publications
********************************************************************************

use "./pub_database", clear
gduplicates drop id_name1 fyear, force
keep id_name1 id_name fyear npub_id_name_fyear sample 

merge m:1 id_name sample using "./dyn_match_all.dta"
keep if _m==3
drop _m

* Identify the appropriate permno_adj to assign the publications to in each year
gen permno_adj=.
forvalue i=0/5 {
replace permno_adj = permno_adj`i' if permno_adj`i'~=. & fyear >= fyear`i' & fyear <= nyear`i'
}
drop if permno_adj == .
keep permno_adj fyear npub_id_name_fyear

* Sum over multiple id_names to get publications for each permno_adj-fyear
gsort permno_adj fyear
gcollapse (sum) npub_id_name_fyear, by(permno_adj fyear)

fillin permno_adj fyear
replace npub_id_name_fyear=0 if npub_id_name_fyear==.

ren npub_id_name_fyear pub_fyear
drop _fillin

* Note that the file is balanced by firm-year (1980-2021) i.e., it contains years when the firm is not traded. These will drop when merged into the firm panel file.

label var pub_fyear "Flow of publications per permno_adj-year"
label var permno_adj "UO firm unique ID"
save "./pub_per_fyear_permno_adj", replace