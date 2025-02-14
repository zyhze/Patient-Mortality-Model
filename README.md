# Mortality Modelling: Hospitalised COVID-19 Patients in Brazil

## Context

This project involves analyzing a dataset of hospitalised COVID-19 patients in Brazil to assess mortality and health risks. The dataset, `CovidHospDataBrasil.csv`, contains information on 157,209 individuals admitted to hospitals between January 1st and December 31st, 2021. The data includes patient demographics, clinical symptoms, pre-existing comorbidities, and vaccination status.

## Project Overview

1. **Modeling Relationship:** Built a model to understand the relationship between patient characteristics and COVID-19 deaths. Used direct comparison methods, regularization or tree methods, and considered spline/polynomial implementations.
2. **Predictive Model:** Predictive model to determine the likelihood of a newly admitted patient dying from COVID-19 based on admission information. Used various methods such as logistic regression, k-nearest neighbours, and classification trees.
3. **Predictive Performance:** Evaluated the accuracy of predictions on evaluation data.
4. **Presentation and Communication:** Ensured the report is well-structured, concise, and easy to read.

## Data Description

The dataset includes:
- **Variables known upon admission:** Age, sex, vaccination status, clinical symptoms (e.g., fever, cough), and pre-existing comorbidities (e.g., diabetes, asthma).
- **Variables known upon death or discharge:** ICU admission, acute respiratory distress syndrome, cardiovascular diseases, and more.
