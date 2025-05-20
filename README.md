# Borosilicate-Glass-Potential

This repository provides example code for running our proposed optimization workflow on molecular dynamics (MD) datasets. Two optimization methods are demonstrated: **Bayesian Optimization** and **CMA-ES**.

---

## 📁 Folder Structure

- **`Bayesian Optimization Example Code.ipynb`**  
  Contains the complete workflow implementation using **Bayesian Optimization** to select parameter sets based on MD simulation results.

- **`CMA-ES Example Code.ipynb`**  
  Demonstrates how to use **CMA-ES** as the optimization method in our workflow. The MLP training process is skipped here, as it is identical to that shown in the Bayesian Optimization example.

- **`data/MD_Dataset.xlsx`**  
  Includes a small portion of our initial MD dataset as an example.  
  *To access the full dataset, please contact the authors.*

- **`utils/`**  
  Contains all utility functions used in the notebooks, including data processing, optimization interfaces, and evaluation tools.

- **`model/`**  
  Provides example trained MLP models used in the optimization workflows:
  - `BO_Model_weight.pth` for Bayesian Optimization  
  - `CMA_Model_weight.pth` for CMA-ES  
  These models allow testing the optimization code without needing to retrain the MLP.

---

## 📝 Notes

All example notebooks are designed to be run in **Google Colab**.  
This repository is for demonstration purposes. For scientific use, please refer to the full dataset and contact us if needed.
