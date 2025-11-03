# LSTM Time Series Forecasting

A complete implementation of Long Short-Term Memory (LSTM) neural networks for time series forecasting using the classic airline passengers dataset.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Keras](https://img.shields.io/badge/Keras-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📊 Overview

This project demonstrates how to use LSTM networks to forecast time series data. The implementation uses the famous airline passengers dataset to predict future passenger counts based on historical trends.

### Key Features

- **Data Preprocessing**: Automatic data loading and normalization
- **LSTM Architecture**: Simple yet effective 4-unit LSTM network
- **Evaluation Metrics**: Root Mean Squared Error (RMSE) calculation
- **Visualization**: Clear plotting of actual vs predicted values
- **Well-Documented**: Step-by-step explanations in the notebook

## 🎯 Results

The model achieves strong performance on the airline passengers dataset:
- **Training RMSE**: ~20-25 passengers
- **Test RMSE**: ~45-50 passengers

The visualization shows how well the LSTM captures seasonal patterns and trends in passenger data.

## 📋 Requirements

- Python 3.8+
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/jugalmodi0111/LSTM-TimeSeries.git
cd LSTM-TimeSeries
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Notebook

Open and run the Jupyter notebook:

```bash
jupyter notebook LSTM-TimeSeries.ipynb
```

Or use VS Code with the Jupyter extension to run the notebook interactively.

## 📚 Project Structure

```
.
├── LSTM-TimeSeries.ipynb    # Main notebook with complete implementation
├── README.md                # This file
└── requirements.txt         # Python dependencies
```

## 🔍 What's Inside

The notebook covers the following steps:

1. **Introduction to Time Series & LSTM**
   - Understanding time series forecasting
   - What makes LSTM special for sequential data

2. **Data Loading & Preprocessing**
   - Loading airline passengers dataset from remote URL
   - Data normalization using MinMaxScaler
   - Train/test splitting (67%/33%)

3. **Data Preparation for LSTM**
   - Creating time-lagged sequences
   - Reshaping data to LSTM input format [samples, time steps, features]

4. **Model Building**
   - Sequential LSTM architecture
   - 4 LSTM units with Dense output layer
   - Adam optimizer with MSE loss

5. **Training**
   - 100 epochs of training
   - Batch size of 1 for online learning

6. **Evaluation & Prediction**
   - Making predictions on train and test sets
   - Inverse scaling to original values
   - RMSE calculation

7. **Visualization**
   - Comprehensive plot showing:
     - Actual passenger data
     - Training predictions
     - Test predictions

## 🎓 Learning Objectives

By working through this notebook, you'll learn:

- How to prepare time series data for LSTM models
- Building and training LSTM networks with Keras
- Evaluating time series forecasting models
- Visualizing predictions vs actual values
- Best practices for sequence prediction

## 📖 Dataset

The project uses the **International Airline Passengers** dataset:
- **Period**: 1949-1960 (12 years)
- **Frequency**: Monthly
- **Size**: 144 observations
- **Source**: [GitHub - amankharwal/Website-data](https://raw.githubusercontent.com/amankharwal/Website-data/master/airline-passengers.csv)

## 🛠️ Technology Stack

- **Deep Learning**: TensorFlow/Keras
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib
- **Machine Learning**: Scikit-learn
- **Environment**: Jupyter Notebook

## 📈 Model Architecture

```
Model: Sequential
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
LSTM                         (None, 4)                 96        
_________________________________________________________________
Dense                        (None, 1)                 5         
=================================================================
Total params: 101
Trainable params: 101
Non-trainable params: 0
```

## 🔧 Customization

You can modify the following parameters to experiment:

- `look_back`: Number of previous time steps to use (default: 1)
- `LSTM units`: Number of LSTM neurons (default: 4)
- `epochs`: Training iterations (default: 100)
- `batch_size`: Samples per gradient update (default: 1)
- `train/test split`: Ratio of training data (default: 0.67)

## 📝 Notes

- The model uses a simple architecture suitable for learning purposes
- For production use, consider:
  - Adding more LSTM layers
  - Implementing dropout for regularization
  - Using cross-validation
  - Trying different look-back periods
  - Experimenting with bidirectional LSTMs

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset source: [Aman Kharwal's Website Data Repository](https://github.com/amankharwal/Website-data)
- Tutorial inspiration: [Time Series with LSTM in Machine Learning](https://amanxai.com/2020/08/29/time-series-with-lstm-in-machine-learning/)
- Keras/TensorFlow documentation and community

## 📧 Contact

**Jugal Modi**
- GitHub: [@jugalmodi0111](https://github.com/jugalmodi0111)

## ⭐ Star This Repository

If you find this project helpful, please consider giving it a star! It helps others discover the project.

---

*Built with ❤️ for the machine learning community*
