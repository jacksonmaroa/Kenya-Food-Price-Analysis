# Kenya maize and bean prices (WFP data)

Python analysis of retail maize and bean prices in Kenya: unit cleaning, outlier flagging, seasonality, a regional comparison and a forecast tested against simple baselines.

**Data:** WFP "Kenya - Food Prices" on the Humanitarian Data Exchange (HDX): https://data.humdata.org/dataset/wfp-food-prices-for-kenya  
The raw file is not included. Download it and save it as `data/raw/wfp_kenya_food_prices.csv`. Credit: World Food Programme.

## Question
How do retail maize and bean prices move over the year and across regions, and can a simple model forecast them better than "same as last month"?

## Method
1. **Standardise units.** Prices come per KG, 90 KG sack, 200 G, 500 ML and so on. Each was converted to KES per kg or per litre (`to_base` in `src/analyze.py`). 646 rows have units that are not a weight or volume (bunch, head, unit) and were left out.
2. **Keep the right rows.** Retail maize grain and beans only. Maize flour and wholesale prices were removed (4,548 wholesale grain rows). That leaves 4,107 retail rows from 198 markets.
3. **Flag outliers, don't delete blindly.** A market-month is flagged if it is more than double or less than half of the 7-month median of the same market. Only 1 market-month was flagged; it is saved in `outputs/flagged_outliers.csv`.
4. **Use the same markets throughout.** Retail maize and bean reporting grew from 4 markets in 2014 to 170 in 2024, so a national average would show who reported, not prices. I used a fixed panel of markets that reported in at least 80% of months from 2015-01 to 2020-12 (8 for maize, 4 for beans). Each market is scaled so its own average is 100, then averaged.
5. **Forecast.** Ridge regression on the last 1, 2, 3 and 12 months plus calendar month, trained before 2020 and tested on 12 held-out months of 2020, one step ahead. Compared with "same as last month" and "same as a year ago".

## Findings
| | |
|---|---|
| Maize seasonality (panel) | Highest in May, lowest in September, about a 6.7% swing |
| Bean seasonality (panel) | Highest in July, lowest in February, about an 8.2% swing |
| Median retail maize since Sep 2024 | Rift Valley 62, North Eastern 70, Eastern 76, Coast 80 KES/kg |
| Forecast error, 12 held-out months | Ridge 2.53, last month 1.89, same month last year 5.93 (index points) |

**The model did not beat the simple baseline.** Predicting "same as last month" had a lower error than Ridge, and 2020 was an unusual year. A sensible conclusion is that month-to-month prices here are very persistent and more data would be needed before trusting a model.

## Charts
`outputs/01_price_trends.png`, `02_seasonality.png`, `03_market_comparison.png`, `04_forecast.png`

## Run it
```
pip install -r requirements.txt
python src/analyze.py --data data/raw/wfp_kenya_food_prices.csv --start 2015-01 --end 2020-12 --compare-start 2024-09 --compare-min-months 1
```

## Limitations
- The panel is small (8 markets for maize, 4 for beans) and mostly in arid and semi-arid counties (Garissa, Kajiado, Kilifi, Kitui, Turkana, Mandera, Baringo, Marsabit), so results are not national.
- Seasonality uses 6 years, which includes the 2017 price spike and 2020.
- Regional medians since Sep 2024 mix many markets and varieties, so they show rough differences, not exact ones.
- Prices are nominal (not adjusted for inflation).
