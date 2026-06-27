# 🏦 Banking Subscription Prediction & Demographic Data Analysis
*March 2024 - April 2024*

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

## 📌 Project Overview
This repository contains the end-to-end machine learning lifecycle for predicting financial product subscriptions. By designing robust data pipelines, I streamlined the analysis of **150,000+ banking and demographic records** to uncover hidden customer behavior patterns and identify high-conversion target demographics. 

The insights derived from extensive feature engineering and exploratory data analysis (EDA) were used to optimize marketing strategies and train powerful ensemble machine learning models, ultimately achieving **~87% prediction accuracy**.

---

## 📊 1. Exploratory Data Analysis & Target Demographics
To drive product strategy, we first needed to understand the underlying behavioral and demographic distributions of our client base.

### Target Variable: Financial Product Subscription
> ![Deposit Distribution](image_701663.png)
* **Insight:** The dataset is highly imbalanced, with the vast majority of clients declining the subscription. Identifying the niche demographics of the "yes" class is the primary objective of our data pipeline.

### Demographic Deep Dive
Understanding the occupational, marital, and educational backgrounds of the target audience.
* **Occupational Landscape:** Admin, Blue-collar, and Technician roles dominate the contact list. 
  > ![Job Bar Plot Blue](image_7019eb.png) 
  *(Alternate view: ![Job Bar Plot Yellow](image_7015c9.png))*
* **Marital Status:** Over 50% of targeted users are married.
  > ![Marital Bar Plot Blue](image_701a08.png)
  *(Alternate view: ![Marital Bar Plot Yellow](image_7015e3.png))*
* **Educational Background:**
  > ![Education Bar Plot](image_7015e7.png)

### Continuous Variables & Macroeconomic Indicators
Analyzing the distribution of age, call duration, and systemic economic conditions (employment variation, Euribor rates) that parallel the campaign.
> ![Grid of Histograms](image_701323.png)

---

## 💰 2. Financial & Campaign Feature Engineering
Analyzing existing client liabilities and marketing engagement metrics to engineer predictive features.

**Financial Liability (Housing vs. Personal Loans):**
> ![Housing Loan](image_701609.png) | ![Personal Loan](image_701622.png)
* **Insight:** Clients predominantly hold housing assets/loans but avoid liquid personal debt, indicating a financially conservative demographic prime for stable term deposits.

**Campaign Timing & Execution:**
> ![Month of Contact](image_701640.png) | ![Day of Week](image_701645.png) | ![Contact Channel](image_701627.png)
* **Insight:** Outreach spiked in May. Daily engagement was consistently distributed across the standard Monday-Friday workweek, with a strong preference for cellular communication.

---

## 🔬 3. Advanced Statistical Analysis & Data Pipelines
To prepare the data for our ensemble ML models, rigorous statistical validation, outlier detection, and correlation mapping were performed.

### Outlier Detection & Feature Dispersion
Identifying skewness in call durations and extreme age outliers to prevent model overfitting.
> **Standard Boxplots:** ![Boxplots Brown](image_7016e2.png) 
> **Alternative Styling:** ![Boxplots Olive](image_701986.png)

### Multicollinearity & Feature Relationships
Mapping pairwise relationships to identify overlapping features (e.g., highly correlated macroeconomic indicators) and isolate independent variables.
> **Pairplot (Colored by Default Status):** 
> ![Pairplot](image_701a25.png)

> **Full Correlation Heatmap:**
> ![Correlation Heatmap](image_701a29.png)

> **Macroeconomic Correlation Focus:**
> ![Filtered Correlation](image_70198b.png)
* **Insight:** Extreme positive correlation (~0.97) between the Euribor 3-month rate and the employment variation rate necessitated careful feature selection during the pipeline engineering phase.

---

## 🤖 4. Machine Learning Architecture: Ensemble Models
Developed and optimized ensemble ML models to achieve robust predictive performance. Below is the visualized logic of the Decision Tree architecture (a core base estimator for our ensemble approach).

### Decision Tree Logic & Node Splitting
The algorithm dynamically routes clients based on optimally reduced entropy, utilizing features like macroeconomic indices (`x[10]`, `x[12]`) and contact metrics to separate converters from non-converters.

> **Granular Node View (Topological Split):**
> ![Zoomed Decision Tree](image_7019c6.png)

> **Full Tree Architecture (Depth & Complexity):**
> ![Full Decision Tree](image_7019ad.png)

### Performance Metrics
* **Accuracy:** ~87% 
* **Optimization:** Hyperparameter tuning was applied across the ensemble structure to balance precision and recall, ensuring the model effectively identifies true subscribers despite the heavily imbalanced dataset.

---

## 🛠️ Technologies Used
* **Programming Language:** Python
* **Machine Learning:** Scikit-learn (Decision Trees, Ensemble Models)
* **Data Engineering/Pipelines:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn

---
*Developed by [Mohammed Ridhwan](https://github.com/MdRidhwan5)*
