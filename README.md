# UKCOVID-CityLevelOutbreakModels

A quick and straightforward framework to perform short-term forecasting when only Total Case data (i.e. cumulative number of positively tested cases) is available. This is the case for UK local authority-level data, which are only reporting cumulative case counts, collated [here](https://github.com/tomwhite/covid-19-uk-data). We basically follow the methodology outlined in this paper: [Short-term Forecasts of the COVID-19 Epidemic in Guangdong and Zhejiang, China](https://www.mdpi.com/2077-0383/9/2/596).

There is a Model class which provides a general framework to specify a function that describes case growth, fit the data, bootstrap confidence intervals, visualise forecasts, and assess performance through RMSE. It is very bare-bones as meant to be exploratory/a useful starting point for a more detailed analysis.

Documenting is a little on the sparse side as I've been hit by a flu myself, so apologies if anything is unclear. The Trends Notebooks should outline how the class is used, however. Please do contact me for details if needed.
