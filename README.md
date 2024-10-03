# World Happiness Analysis

## Overview
This project explores the [`world_happiness.csv`](/data/world_happiness.csv) dataset, which contains various variables that serve as proxies for evaluating the quality of life in different countries. The primary focus is on understanding the factors influencing happiness scores across different regions.

## Dataset
The dataset can be found in the `/data` folder. It includes several variables, including:

- **happiness_score**: A metric summarizing how "happy" each country is.
- Other factors such as social support, freedom, corruption, and life expectancy.

## Analysis Steps
1. **Data Loading and Exploration**: Load the dataset into a Pandas DataFrame and examine its structure.
2. **Data Cleaning**: Identify and handle missing values, filling them with appropriate averages based on regional data.
3. **Data Visualization**:
   - Plot the distribution of the `happiness_score` variable using a histogram.
   - Create a correlation heatmap to visualize relationships between numeric variables.
4. **Statistical Modeling**:
   - Develop a linear regression model using `life_exp` to predict `happiness_score`.
   - Evaluate model performance using R-squared and Mean Absolute Error (MAE).
   - Assess potential overfitting by comparing R-squared values for training and test datasets.

## Key Findings
- The distribution of happiness scores resembles a continuous uniform distribution with minor deviations at the extremes.
- The variable least correlated with `happiness_score` is identified, and significant relationships are explored through correlation analysis.
- The analysis shows that a 1-year increase in life expectancy corresponds to an increase of approximately 5.10 points in the happiness score.

## Code
All calculations and analyses are implemented in the Python file located in the [`global_happiness_insights.py`](/data/global_happiness_insights.py) in `/data` folder. The Python code handles data loading, cleaning, visualization, and modeling.
