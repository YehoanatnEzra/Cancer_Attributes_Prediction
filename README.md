# Breast Cancer Attributes Prediction

Imagine dropping a patient’s record into a command line and, moments later, learning not only the chance that cancer is present but also the subtype most likely to emerget

This project is a 48-hour group effort from the IML Hackathon 2025. It implements a complete machine-learning pipeline that begins by gathering and sanitising raw clinical data—laboratory results, demographic details, surgical summaries, and more—and transforms them into a consistent, information-rich feature set. On this foundation we trained a model that, given the medical profile of a single patient, instantly returns the probability that the patient currently has cancer and when risk is high, the most likely cancer subtype or metastasis pattern. Every stage of the workflow from data ingestion and preprocessing through model inference and formatted output runs automatically, delivering clear, clinically relevant predictions without any manual intervention.

Behind the scenes, out pipeline combines three modelling stages: a class-balanced Random Forest (200 trees) for multi-label metastasis prediction, a Gradient-Boosting Regressor with quantile loss for tumour-size estimation, and an unsupervised stack of PCA → t-SNE → k-means (k = 5) for phenotype discovery. Every step from data ingestion and preprocessing through model inference and formatted output runs automatically, delivering clear, clinically relevant predictions without manual intervention. All findings and visualisations from the clustering stage are summarised in taskIII/project.pdf, ready for research or presentation.

## Repository Contents

- requirements.txt - Core Python dependencies for the full pipeline. 
- load_and_split.py - Loads raw data and enforces patient-level train/val splits. 
- process_data.py - Cleans columns, imputes missing values, quantises features. 
- labels_utils.py - Converts metastasis labels to/from multi-hot vectors. 
- subtaskI/ - Code, requirements and predictions for Task I (metastasis). 
- part1.py - Trains the Random-Forest model and writes predictions. 
- predictions.csv - Pre-computed Task I predictions on test features. 
- subtaskII/- Analogous structure for Task II (tumour size). 
- subtaskIII/ - Unsupervised analysis notebook, PDF report and helper code. 


## Feedback & Contact
If you find any issues, have questions, or suggestions for improvement, feel free to reach out:

- Yehonatan Ezra - https://www.linkedin.com/in/yehonatanezra/  
- Lior Zats - https://www.linkedin.com/in/lior-zats/
- Avi Kfir  - https://www.linkedin.com/in/avi-klings/
- Shay Morad - https://www.linkedin.com/in/shay-morad/

## License
This project is open-source and available for personal and educational use.
