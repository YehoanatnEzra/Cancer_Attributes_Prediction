# iml-hack-oncology
Our hackathon codebase constitutes a self-contained clinical machine-learning pipeline that converts heterogeneous hospital records into reliable cancer-related predictions within minutes. The workflow begins with a comprehensive preprocessing stage: each raw CSV is parsed, duplicate visits are merged, Hebrew and English free-text entries are standardised, categorical variables are one-hot encoded or embedded, and missing values are handled by K-nearest-neighbour imputation for numeric fields and a dedicated “missing” category for categoricals. Consequently, every patient visit is transformed into a consistent, information-rich feature vector that integrates temporal patterns, biological markers, and surgical history.

Building on this foundation, the repository offers three modelling components.

1. **Metastasis prediction** – a class-balanced Random Forest with 200 trees delivers multi-label forecasts of metastasis sites; calibrated decision thresholds achieve a micro-F1 score of approximately 0.78 on cross-validation.
2. **Tumour-size estimation** – a Gradient-Boosting Regressor trained with quantile loss remains robust to the heavy-tailed distribution of tumour sizes, attaining a mean-squared error near 32 mm².
3. **Unsupervised phenotype discovery** – dimensionality reduction via PCA followed by t-SNE and k-means clustering (k = 5) uncovers clinically coherent patient groups that align with recognised subtypes such as HER2-positive and triple-negative disease.

The entire workflow is orchestrated by lightweight MLOps utilities: executing `make all` creates isolated virtual environments, runs each task sequentially, logs key metrics, serialises trained models, and packages the outputs—three prediction files plus a manifest—into the exact ZIP format required by the hackathon leaderboard. Re-running the same command on a standard laptop faithfully reproduces the full pipeline, from raw records to probability scores, tumour-size predictions, and cluster visualisations, in under ten minutes, thereby demonstrating complete ownership of data engineering, machine-learning modelling, and automated deployment.


