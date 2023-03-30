# Breast-Cancer-Detection-
BUSINESS OBJECTIVE: Predicting whether a tumor is benign or malignant based on certain features. This can assist medical professionals in making more informed decisions about patient care and treatment options. Additionally, it can aid in the early detection of breast cancer, which is crucial for improving patient outcomes and survival rates.

Feature Information:

* Clump Thickness: Clump thickness is one of the morphological features of breast cells that is often evaluated during the diagnosis of breast cancer. The thickness of the cell clusters or cell aggregates in breast tissue samples is considered to be an important factor in the determination of cancerous versus non-cancerous cells.A higher value for clump thickness suggests that the cells are growing and dividing more rapidly, and that there is a greater likelihood that they are cancerous. In contrast, a lower value for clump thickness may indicate a less aggressive form of cell growth that is less likely to be cancerous.

* Size Uniformity: Size uniformity is a feature in a breast cancer dataset that refers to the uniformity or consistency in the size of the nuclei of breast cells observed in a tissue sample from a breast biopsy.High values for size uniformity are generally associated with a higher likelihood of cancerous growth, as the variability in the size of nuclei can be indicative of abnormal cell growth and cell division. In contrast, low values for size uniformity suggest that the cells in the tissue sample are relatively uniform in size and shape, which is more consistent with normal, healthy breast tissue.

* Shape Uniformity: Shape uniformity is a feature in a breast cancer dataset that refers to the degree of variation in the shape of cells in a tissue sample obtained from a breast biopsy. More specifically, it assesses the degree of uniformity or variation in the size and shape of the nuclei of the cells in the tissue sample.

* Marginal Adhesion: Marginal adhesion is a measure of how well the cancerous cells in a tissue sample stick together. It refers to the extent to which the cancer cells are able to attach to each other and to the surrounding tissues. In other words, it measures the degree to which the cancer cells are able to form cohesive groups or clusters. Marginal adhesion is measured on a scale of 0 to 10, where 0 indicates that the cells are not adherent at all and 10 indicates that the cells are highly adherent. This measurement is typically made by examining the tissue sample under a microscope and evaluating the degree to which the cells are clustered together.

* Epithelial Size: Epithelial size is a feature in the breast cancer detection dataset that measures the size of the epithelial cells present in a tissue sample. The epithelium is the outer layer of cells that covers the surface of the tissue, and changes in its size and shape can be an indicator of cancer. Epithelial size is measured on a scale of 1 to 10, with 1 being the smallest and 10 being the largest. Higher scores indicate larger epithelial cells, which may be indicative of cancer.

* Bare Nucleoli: Bare_nucleoli refers to the appearance of nuclei in a stained histologic section of a cell. It is a measure of the appearance and distribution of the nuclei in the cell. Normally, nucleoli are only visible within the nucleus of a cell, but when a cell is cancerous, the nucleoli may become more visible and may be present outside the nucleus. This feature can be used to help diagnose cancer and determine its severity.

* Bland Chromatin: Bland chromatin refers to the appearance of the chromatin (the material that makes up chromosomes) within the nucleus of a cell. It is a measure of how uniform and smooth the chromatin looks.In cancer diagnosis, bland chromatin is often evaluated to help determine whether a tumor is benign or malignant. In general, benign tumors have bland chromatin, meaning that the chromatin appears uniform and smooth, while malignant tumors have irregular chromatin that is clumped together and may appear darker or more variable in shape and size.

* Normal Nucleoli: Normal_nucleoli is a feature in the breast cancer detection dataset that refers to the size and shape of the nucleoli (small structures within the nucleus of a cell) present in the cell nuclei of a breast tissue sample. Abnormalities in the size and shape of nucleoli can be an indicator of abnormal cell growth and division, which is a hallmark of cancer.

* Mitoses: Mitosis is a process of cell division that occurs when a cell divides into two identical daughter cells. In normal tissue, mitosis occurs at a controlled rate, but in cancerous tissue, mitosis can occur at an accelerated rate leading to the formation of abnormal cells.

Conclusion:

* In health care sector precision is extremely importance hence to choose best model we have consider F1 score along with accuracy.
* It is always good practice to use Ensemble techniques because we are not giving conclusion from just one model,we are building multiple models and giving conclusion based on majority voting.
* Random forest is a ensemble technique and one of the advantage of using random forest is,it significantly reduces overfit problem.
* Naive Bayes has slighly more F1 score than Random forest but accuracy is same for both the models.
* Random Forest could be a good choice from all the models that I have tried.

Future Scope:

* If all features in the dataset had balanced values across all the classes, it would have been easier to analyze and identify trends and patterns within the data. The current imbalance in class distribution limits our ability to draw meaningful insights from the data, as the models may be biased towards the majority class.
* I have not worked on imbalance of data. We can balance it and check how accuracy is affecting.
* I did not use a deep learning model in this analysis due to the high computational requirements of these models. However, we could explore using a deep learning model to see if it improves the accuracy of our predictions.
