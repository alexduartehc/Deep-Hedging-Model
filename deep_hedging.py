import numpy as np
import matplotlib.pyplot as plt
import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import losses
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, Subtract, BatchNormalization, Activation, Flatten, MaxPooling2D, Multiply, Lambda, Add, Dot

#if not tf.__version__.split(".")[0] == "1":
#    import tensorflow_probability as tfp
#    tfd = tfp.distributions

from stochastic_processes.gbm import GBM
from black_scholes.bs_functions import BS

class DeepHedging():
    def __init__(self, parameters):

        # Set parameters from parameters file
        self.paths = parameters["simulation"]["paths"]
        self.n = parameters["simulation"]["n"]
        self.cost = parameters["hedging"]["cost"]
        self.cost_type = parameters["hedging"]["cost_type"]
        self.epsilon = parameters["hedging"]["epsilon"]    # used for fixed cost
        self.initial_hedge = parameters["hedging"]["initial_hedge"]
        self.alpha = parameters["hedging"]["alpha"]
        self.loss_type = parameters["hedging"]["loss_type"]
        self.S = parameters["option"]["S"]
        self.K = parameters["option"]["K"]
        self.r = parameters["option"]["r"]
        self.sigma = parameters["option"]["sigma"]
        self.div = parameters["option"]["div"]
        self.T = parameters["option"]["T"]
        self.h = parameters["option"]["h"]
        self.prices = parameters["simulation"]["training_prices"]
        self.test_prices = parameters["simulation"]["test_prices"]
        self.d = parameters["nn"]["d"]
        self.l = parameters["nn"]["l"]
        self.epochs = parameters["nn"]["epochs"]
        self.batch_size = parameters["nn"]["batch_size"]

        self.payoffs = self.get_payoffs(parameters["simulation"]["training_prices"])
        self.priceBS = BS(parameters["option"]["S"], parameters["option"]["K"], parameters["option"]["T"], parameters["option"]["sigma"], parameters["option"]["r"], parameters["option"]["div"],
                          parameters["option"]["lambd"], parameters["option"]["mu_J"], parameters["option"]["sigma_J"], N=50)
        self.model = self.create_models(self.d, self.l)
        self.current_timestep = 0
        self.xtrain = None
        self.ytrain = None

    # Builtin function that returns the predicions for a set of prices
    def __call__(self, prices):
        payoffs = self.get_payoffs(prices)
        x, y = self.create_trainingset(prices, payoffs)
        pred = -self.model.predict(x)[:,0]
        return pred

    # Create trainingset based on prices and payoffs
    def create_trainingset(self, prices, payoffs):
        differences = np.diff(prices)
        xtrain = [np.expand_dims(prices[:, 0], -1)]
        for i in range(differences.shape[1]):
          xtrain.append(np.expand_dims(differences[:,i], -1))
        ytrain = np.zeros_like(payoffs)
        return xtrain, ytrain

    # Custom MSE loss function
    def custom_mse(self, y_true,y_pred):
        z = y_pred[:,0]-y_true[:,0]
        z=K.mean(K.square(z))
        return z

    # Custom CVaR loss function
    def custom_cvar(self, y_true, y_pred):
        CVaR, idx = tf.nn.top_k(tf.squeeze(y_pred), tf.cast((1-self.alpha)*tf.cast(tf.size(y_pred), tf.float32), tf.int32))
        CVaR = tf.reduce_mean(CVaR)
        return CVaR

    # Custom Quadratic CVaR loss function
    def custom_cvar_square(self, y_true, y_pred):
        CVaR, idx = tf.nn.top_k(tf.squeeze(y_pred), tf.cast((1-self.alpha)*tf.cast(tf.size(y_pred), tf.float32), tf.int32))
        CVaR = tf.reduce_mean(tf.square(CVaR))
        return CVaR

    # This function creates the neural networks for each time step and connects the
        # to create the deep hedging model.
    def create_models(self, d, l):
        models = []
        price = Input(shape=(1,))
        old_hedge = tf.constant(self.initial_hedge, shape=(1,), dtype=tf.float32)
        wealth = tf.constant(0, shape=(1,), dtype=tf.float32)
        epsilon = tf.constant(self.epsilon, shape=(1,), dtype=tf.float32)

        inputs = [price]

        # Create a model for each time step
        for i in range(self.n):
            x = Dense(l, name="first_"+str(i))(price)
            x = Activation("tanh")(x)
            x = Dense(1, name="second_"+str(i))(x)
            hedge = Activation("linear")(x)

            price_diff = Input(shape=(1,))
            x = Multiply(name="hedge_times_increment"+str(i))([hedge, price_diff])

            if self.cost_type == "constant_argmax":
                hedge_change = Lambda(lambda x: tf.abs(x[0]-x[1]), name ="hedge_change"+str(i))([hedge, old_hedge])
                temp = Lambda(lambda x: (tf.sign(epsilon - x)-1)*(-1/2))(hedge_change)
                cost = Lambda(lambda x: self.cost*x)(temp) # c if hedge_change > epsilon
                x = Subtract(name="cost"+str(i))([x, cost])

            elif self.cost_type == "proportional":
                hedge_change = Lambda(lambda x: tf.abs(x[0]-x[1]), name ="hedge_change"+str(i))([hedge, old_hedge])
                cost = Lambda(lambda x: self.cost*x[0]*x[1])([price, hedge_change]) # c_k from paper c*S*|n|
                x = Subtract(name="cost"+str(i))([x, cost])

            wealth = Add(name="wealth"+str(i))([x, wealth])
            inputs = inputs + [price_diff]
            price = price + price_diff
            old_hedge = hedge

        payoff= Lambda(lambda x : 0.5*(tf.abs(x-self.K)+x-self.K))(price)
        x = Subtract(name="payoff")([payoff, wealth])
        output = Subtract(name="premium")([x, tf.constant(self.priceBS, shape=(1,), dtype=tf.float32)])

        model = Model(inputs=inputs, outputs=output)
        opt = tf.keras.optimizers.SGD(learning_rate=0.002)
        if self.loss_type == 'mse':
            model.compile(optimizer=opt, loss=self.custom_mse)
        elif self.loss_type == 'cvar':
            model.compile(optimizer=opt, loss=self.custom_cvar)
        elif self.loss_type == 'cvar_square':
            model.compile(optimizer=opt, loss=self.custom_cvar_square)

        return model

    # Train the deep hedging model and return the resulting PnL based on test prices
    def train_models(self):
        self.xtrain, self.ytrain = self.create_trainingset(self.prices, self.payoffs)

        self.model.fit(x=self.xtrain, y=self.ytrain, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2, verbose=True)#, callbacks=[tensorboard_callback])

        self.xtest, _ = self.create_trainingset(self.test_prices, self.payoffs)
        pnl = -self.model.predict(self.xtest)[:,0] # - to get true pnl
        #plt.hist(pnl, bins=50)
        #plt.show()
        print("Mean: ", np.mean(pnl))
        print("Std: ", np.std(pnl))
        return pnl

    # Get vanilla European option payoffs based on prices and strike prices
    def get_payoffs(self, prices):
        return np.maximum(prices[:, -1] - self.K, 0)

    # Return the final PnL based on the trained model
    def get_final_pnl(self, prices):

        xtest, ytest = self.create_trainingset(prices, self.get_payoffs(prices))

        return -self.model.predict(xtest)[:,0] # - to get true pnl

    #
    def deltastrategy(self, s, time_to_maturity):


    # 1. Rebuild a sub-model that reproduces the hedge network at timestep `time_to_maturity`
        price_input = Input(shape=(1,), name='price_input')
        x = Dense(self.l, name=f"first_{time_to_maturity}")(price_input)
        x = Activation("tanh")(x)
        x = Dense(1, name=f"second_{time_to_maturity}")(x)
        hedge_output = Activation("linear")(x)

        delta_model = Model(inputs=price_input, outputs=hedge_output)

    # 2. Assign weights from the full model to the sub-model
        full_weights = dict(zip(
            [w.name for w in self.model.weights],
            self.model.get_weights()
            ))

        for layer in delta_model.layers:
            weights = []
            for weight in layer.weights:
                if weight.name in full_weights:
                    weights.append(full_weights[weight.name])
            if weights:
                layer.set_weights(weights)

    # 3. Predict the hedge using the reconstructed sub-model
        s = np.asarray(s).reshape(-1, 1).astype(np.float32)
        hedge = delta_model.predict(s, verbose=0).flatten()
        return hedge

