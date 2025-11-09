# Deep-Hedging-Model

This repository implements a Deep Hedging framework. It is a neural network–based approach to dynamic risk management and option hedging under realistic market conditions.
Inspired by Josef Teichmann's work (2019), this implementation uses TensorFlow to model and optimize hedging strategies that account for transaction costs, market frictions, and nonlinear risks.

Traditional delta hedging assumes perfect liquidity and continuous rebalancing, which rarely hold in practice.
Deep Hedging replaces analytical formulas with data-driven neural networks capable of learning optimal dynamic hedging strategies directly from simulated market data.

This project demonstrates how to:
  * Simulate asset price paths using stochastic processes
  * Train a neural network to minimize portfolio risk or tail losses
  * Incorporate transaction cost models (fixed, proportional, etc.)
  * Evaluate hedging performance using profit and loss (PnL) distributions
  * The model can approximate strategies under different loss functions such as MSE, CVaR, or quadratic CVaR.


Key Concepts:
  * Deep reinforcement learning for financial risk management
  * Stochastic asset price simulation (Geometric Brownian Motion)
  * Neural network–based hedging policy optimization
  * Transaction cost modeling
  * Conditional Value-at-Risk (CVaR) optimization
  * Comparison with analytical Black–Scholes delta hedging


Model Features:
  * Multiple Loss Functions: Mean Squared Error (MSE); Conditional Value-at-Risk (CVaR); Quadratic CVaR
  * Transaction Cost Models: Constant or fixed threshold costs; Proportional to traded notional
  * Dynamic Neural Hedging Policy: The model constructs a sequence of dense neural networks, one per time step, each learning optimal hedge adjustments.


Once trained, the model can output:
  * Hedging PnL distribution histograms
  * Expected return and standard deviation
  * Estimated risk metrics (e.g., CVaR at α = 0.95)
  * Learned hedge ratios across timesteps

--------

Deep-Hedging

│

├── deep_hedging.py                   # Main Deep Hedging model implementation

├── README.md                         # Project documentation

└── LICENSE
