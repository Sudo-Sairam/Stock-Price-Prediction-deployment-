# Stock Price Prediction Model

## Overview

This project presents a machine learning-based stock price prediction system developed to forecast future stock movements using historical market data. It is the result of a structured development process, starting with a working prototype and evolving into a more robust final version.

### Project Focus

- **Performs well with low-volatility stocks:** The model is highly effective at predicting stocks with relatively stable and predictable price patterns.
- **Challenges with high-volatility stocks:** The prediction accuracy decreases when dealing with stocks that show sudden or frequent price fluctuations. This is a recognized limitation and an area identified for future research.

## Technologies Used

- **Programming Language:** Python  
- **Libraries & Tools:**  
  - Machine Learning: `scikit-learn`, `TensorFlow` or `PyTorch`  
  - Data Handling: `pandas`, `NumPy`  
  - Visualization: `matplotlib`, `seaborn`  
  - Data Acquisition: `yfinance`, `Alpha Vantage`

## Features

- Data collection from financial APIs
- Preprocessing and feature extraction
- Training and evaluation of ML models
- Visualization of predictions and performance
- Performance comparison across different stock volatility profiles

## Limitations

- **Volatility Sensitivity:** The model currently underperforms on stocks with high price volatility.
- **External Factors Ignored:** Sudden market movements due to news or global events are not factored into the model.

## Future Work

- Integrate news sentiment analysis to improve high-volatility predictions.
- Experiment with advanced time-series models (e.g., LSTM with attention mechanisms).
- Deploy a real-time dashboard for live predictions.
- Add macroeconomic features to improve robustness.

## Getting Started

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/stock-prediction-model.git
cd stock-prediction-model
pip install -r requirements.txt
python main.py
```

Update the configuration file (`config.py`) to define the stock symbol, date range, and model parameters.

## Contributors

- **Sairam S**  
- **Sabarish RS**  
- **Rohit Kumar G**  
- **Niranjan Shankar**  
- **Reshvnth**

**Institute:** PSG Institute of Technology and Applied Research (PSG iTech)

## License

This project is licensed under the MIT License. See `LICENSE` for more details.
