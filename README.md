# Causal Inference for Extreme Events

Droughts are hydroclimatic anomalies driven by precipitation deficits and increased evapotranspiration, posing an escalating threat under global warming conditions. However, assessing drought risk remains challenging due to the complex interactions between biophysical conditions and human systems, as well as limitations in impact reporting. Furthermore, the impact of drought varies significantly across different sectors because different types of drought affect socio-environmental systems in different ways. Therefore, an approach based on drought impacts is essential for understanding drought risk. 

Although traditional machine learning (ML) has achieved remarkable success in drought prediction, these models are often based on spurious correlations rather than physical mechanisms. Predictive accuracy does not translate into causal understanding. In order to develop actionable policies for climate adaptation, we must transition from prediction based on associations to causal inference.

This challenge focuses on causal inference methods to identify the causes of extreme weather events. Participants will move beyond association in order to estimate the heterogeneous causal effect of climate and environmental factors on the severity of such events.

Participants are invited to explore the dataset and develop causal inference models to improve our understanding of the impact of drought on the agricultural sector.

### 📚 Recommended reading material

- Reading material ([link](https://drive.google.com/drive/folders/18Y-tuNztgH0uY1eFnf34btzWRkrvkzm0))
- Econml [Documentation](https://www.pywhy.org/EconML/)
  -   [econml.dml.LinearDML](https://www.pywhy.org/EconML/_autosummary/econml.dml.LinearDML.html#econml-dml-lineardml)
  -   [econml.dml.CausalForestDML](https://www.pywhy.org/EconML/_autosummary/econml.dml.CausalForestDML.html#cfdml1)

### 🎯 Challenge Objectives

By using causal inference methods, this challenge aims to uncover the underlying causal drivers of extreme events such as drought impacts on the agricultural sector.

Participants are encouraged to investigate one or more of the following research directions:

- How does **soil moisture deficits** influence the likelihood of agricultural drought impacts, and how do these effects vary across **climate types** and/or  **hydrological basins**?”
- How does **meteorological droughts (SPI)** influence the likelihood of agricultural drought impacts, and how do these effects vary across **climate types** and/or  **hydrological basins**?”
- How does **[ENSO](https://www.ncei.noaa.gov/access/monitoring/enso/)** influence the likelihood of agricultural drought impacts, and how do these effects vary across **climate types** and/or  **hydrological basins**?”
- *How does a "**factor**" influence the likelihood of "**agricultural drought impacts**", and how do these effects vary across "**regions**"?”*

### 🗂️ Data

[data_description.md](https://github.com/WinterSchool2026/ch09-causal-inference-extremes/blob/main/data_description.md)

---

### 🚀 Getting Started

1️⃣ Clone the Repository or open in Google Colab

```
cd C:\repos
git clone https://github.com/martsape/ch09-causal-inference-extremes.git
cd ch09-causal-inference-extremes
```


2️⃣ If the repository has been cloned (this step is not necessary in Google Colab) -> Open terminal -> Install dependencies (with .yml file)

```
conda env create -f environment.yml
conda activate causal_ml
```

3️⃣ Explore the data

TASKS:
- Understand the spatiotemporal distribution of drought impacts, potential biases, and the environmental and climate covariates.
- Play with the data, make some maps, conduct an Exploratory Data Analysis (EDA) to understand the structure of the dataset.
- Propose a Directed Acyclic Graph (DAG) based on the covariates from the list, select a treatment, and the outcome. Look for the literature on the topic, and make a simple DAG to start on.

4️⃣ Data filtering for causal inference (hints: notebooks\02_data_filtering_propensity_score.ipynb)

Propensity scores are used to filter out samples with extreme values to satisfy the overlap assumption, which ensures that the treatment and control groups are sufficiently comparable. Removing these units guarantees that every remaining observation had a non-trivial probability of receiving the treatment, thereby reducing bias and improving the stability of the causal effect estimates

TASKS:
- Train a binary logistic regression model with the Treatment as dependent variable, and the confounders from your DAG as independent variables.
- Remove samples that have extreme Propensity scores (probability of the possitive class in the treatment).
- Save the selected samples in a CSV file for later.

5️⃣ Train/test nuisance models (hints: notebooks\03_trained_nuisance_models.ipynb)

Nuisance models are the first-stage machine learning models used to separately predict the treatment assignment and the outcome variable based solely on the observed covariates. By estimating these arbitrary functions (nuisance parameters), the algorithms can isolate the residual variation in the treatment and outcome that is independent of confounding factors, allowing for an unbiased estimation of the causal effect in the final stage.

TASKS:
- Find the best two nuisance models, one to predict the Treatment and one for the Outcome. 
- Look for the best hyperparameters for each model.
- Use the filtered samples from step 4️⃣
- Split the sample data in train, validation, and test using the time dimension.
- Use cross validation (CV)
- Add the regions or heterogenous variable ($X$) in the nuisance models by creating a hot-one enconded or dummy variable.
- Report the accuracy of the models.

6️⃣ Train causal models (hints: notebooks\04_causal_models.ipynb)

Double Machine Learning (DML) is a semiparametric causal inference technique that leverages machine learning to estimate average treatment effects in the presence of measured confounders by capturing complex, nonlinear relationships. It achieves this by using a procedure called cross-fitting to separately predict the treatment and the outcome from the covariates, and then isolates the causal effect by regressing the residuals from these two models to remove confounding bias. 
LinearDML and CausalForestDML are part of the Double Machine Learning (DML) family in the EconML library. They are designed to estimate the Conditional Average Treatment Effect (CATE).*

**LinearDML**: It assumes that the relationship between the features and the treatment effect is linear. It uses ML to remove the influence of confounders from both the treatment and the outcome, and then fits a simple linear regression on the residuals.

**CausalForestDML**: It is non-parametric. It doesn't assume a linear relationship. It can find complex interactions. It uses a Forest of Trees (similar to a Random Forest) to estimate the treatment effect. It first cleans the data using ML. However, instead of a linear regression at the end, it builds many decision trees. Each tree tries to find groups of samples who respond differently to the treatment. It then averages these trees to get the final effect.

TASKS:
- Use the filtered samples from step 4️⃣, the same treatment and outcome, encode the regions
- Make a map and a graph with the representation of T/O for your regions, in total, and across time and space.
- Train a causal machine learning model to model the residual variation in the treatment and outcome. For that, you need your two best models from step 5️⃣
- Compare a ``LinearDML`` model to ``CausalForestDML`` model
- Report ATE and CATE
- Visualize CATE for your regions

7️⃣ Validation / Refutation tests (hints: notebooks\04_causal_models.ipynb)

In causal machine learning frameworks validation strategies, often called refutation tests or robustness checks, are used to stress-test the underlying causal assumptions and ensure the estimated effects are reliable.

The primary validation strategies used are:
  
**Placebo Treatment**: This test involves randomly permuting (shuffling) the treatment variable in the dataset. Because the newly assigned "placebo" treatment has no actual relationship with the outcome, a robust model should return an estimated causal effect that drops to zero.
**Unobserved Common Cause / Sensitivity Analysis / Omitted Variable Test**: This simulates the presence of an unmeasured confounder (omitted variable) that affects both the treatment and the outcome. Sensitivity analyses quantify how much the final causal estimate would change or degrade under various strengths of this unobserved confounding, helping determine if the conclusions would hold up even if they missed a variable.

**Random Common Cause**: A random noise variable is added to the dataset as an additional mock confounder. Because the variable is just noise, the estimated causal effect should remain unchanged and stable

**Random Subset Removal:** A random subset of the observations is removed from the dataset, and the model is re-run. The estimated effect is expected to remain consistent, showing that the results are not being driven by a specific, small cluster of data points

TASKS:
  - Validate the Nuisance models from 5️⃣
  - Apply different validation methods to the causal model.

8️⃣ Interpret results

TASKS:
- Interpret your results. How much your treatment affected the impacts of agricultural droughts in different regions of Europe?
- Look for literature that supports or contradicts your findings and discuss if your findings are plausible.
- Prepare a presentation for Friday with your main findings, the challenges and limitations of the analysis.
