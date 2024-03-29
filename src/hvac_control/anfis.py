import tensorflow as tf

class ANFIS(tf.keras.Model):
    """
    This class implements an Adaptive Neuro-Fuzzy Inference System (ANFIS), which is a kind of artificial neural network 
    The class extends tf.keras.Model, integrating seamlessly with TensorFlow's model structure.
    """
    def __init__(self, 
                 n_inputs, 
                 n_rules, 
                 learning_rate=1e-2,
                 mf='gaussmf',
                 defuzz_method='proportional',
                 loss_fun='mse',
                 init_method='uniform'
                 ):
        super(ANFIS, self).__init__()

        """Initializes an ANFIS model

        Attributes:
        - n_inputs (int): The number of input variables.
        - n_rules (int): The number of fuzzy rules.
        - learning_rate (float): The learning rate for the optimizer. Default is 1e-2.
        - mf (str): The membership function type. Supported values are 'gaussmf' for Gaussian and 'gbellmf' for Generalized Bell. Default is 'gaussmf'.
        - defuzz_method (str): The defuzzification method. Supported values are 'proportional' and 'linear'. Default is 'proportional'.
        - loss_fun (str): The loss function to use. Supported values are 'mse' for Mean Squared Error and 'huber'. Default is 'mse'.
        - init_method (str): The method for initializing variables. Supported values are 'uniform' and 'normal'. Default is 'uniform'.
        """

        self.n = n_inputs
        self.m = n_rules
        self.define_variables(mf, defuzz_method, init_method)

        self.mf = mf
        self.defuzz_method = defuzz_method
        self.loss_fun = loss_fun
        self.init_method = init_method
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def define_variables(self, mf, defuzz_method, init_method):
        """
        Define and initialize the variables for the ANFIS model.

        This method initializes the variables required for the ANFIS model based on the specified 
        membership function, defuzzification method, and initialization method. It creates TensorFlow 
        variables for the parameters of the membership functions and the defuzzification layer.

        Parameters:
        - mf (str): The type of membership function ('gaussmf' for Gaussian or 'gbellmf' for Generalized Bell).
        - defuzz_method (str): The defuzzification method ('proportional' or 'linear').
        - init_method (str): The method for initializing variables ('uniform' or 'normal').
        """

        def initialize_variable(shape, name, min_val=0, max_val=None):
            if init_method == 'uniform':
                return tf.Variable(tf.random.uniform(shape, minval=min_val, maxval=max_val), name=name)
            elif init_method == 'normal':
                return tf.Variable(tf.random.normal(shape, mean=0.0, stddev=.1,), name=name)
            else:
                raise ValueError("Invalid initialization method")

        if mf == 'gaussmf':
            self.mu = initialize_variable([self.m * self.n], "mu", -1.5, 1.5)
            self.sigma = initialize_variable([self.m * self.n], "sigma", .7, 1.3)

        elif mf == 'gbellmf':
            self.a = initialize_variable([self.m * self.n], "a", .7, 1.3)
            self.b = initialize_variable([self.m * self.n], "b", .7, 1.3)
            self.c = initialize_variable([self.m * self.n], "c", -1.5, 1.5)

        if defuzz_method == 'proportional':
            self.y = initialize_variable([1, self.m], "y", -2, 2)

        elif defuzz_method == 'linear':
            #self.coefficients = tf.Variable(tf.random.normal([self.m, self.n]), name="coefficients")
            #self.intercepts = tf.Variable(tf.random.normal([self.n]), name="intercepts") 
            self.coefficients = initialize_variable([self.m, self.m], "coefficients", -2, 2)
            self.intercepts = initialize_variable([self.m], "intercepts", -2, 2)

                
    def call(self, inputs):
        """
        Perform a forward pass of the ANFIS model.

        This method implements the forward pass of the ANFIS model. It computes the membership values 
        for each input using the defined membership functions, calculates the firing strength of 
        each rule, normalizes these strengths, and then applies the defuzzification process to 
        produce the final output.

        Parameters:
        - inputs (Tensor): The input tensor to the model.

        Returns:
        - Tensor: The output tensor of the ANFIS model.
        """

        # Layer 1: membership layer
        if self.mf == 'gaussmf':
            membership_values = tf.exp(-0.5 * tf.square(tf.subtract(tf.tile(inputs, (1, self.m)), self.mu)) / tf.square(self.sigma+1e-12))

        elif self.mf == 'gbellmf':
            abs_val = tf.abs((tf.subtract(tf.tile(inputs, (1, self.m)), self.c)) / (self.a+1e-12))
            membership_values = 1 / (1 + tf.pow(abs_val, 2 * self.b))

        # Reshape the membership values for the rule layer
        reshaped_membership_values = tf.reshape(membership_values, (-1, self.m, self.n))

        # Layer 2: Rule Layer - Calculate the firing strength of each rule
        rul = tf.reduce_prod(reshaped_membership_values, axis=2)

        # Layer 3: Normalization Layer - Normalize the firing strengths
        den = tf.clip_by_value(tf.reduce_sum(rul, axis=1), 1e-12, 1e12)  # Avoid division by zero

        # Layer 4: Defuzzification Layer - Compute the weighted sum of each rule's output
        if self.defuzz_method == 'proportional':
            num = tf.reduce_sum(tf.multiply(rul, self.y), axis=1)  # Weighted sum of the rule outputs

        elif self.defuzz_method == 'linear':

            linear_output = tf.matmul(rul, self.coefficients)

            #reshaped_intercepts = tf.reshape(self.intercepts, (1, self.n))  # Reshape intercepts            
            reshaped_intercepts = tf.reshape(self.intercepts, (1, self.m))  # Reshape intercepts

            linear_output = tf.add(linear_output, reshaped_intercepts)  # Now add
            num = tf.reduce_sum(linear_output, axis=1)

        # Layer 5: Output Layer - Compute the final output
        out = tf.divide(num, den)  # Final output of the ANFIS model

        return out

    def train_step(self, data):
        """
        Perform one training step for the ANFIS model.

        This method overrides the train_step method of tf.keras.Model. It computes the loss for the given 
        data and applies gradients to optimize the model. The loss function used is specified during 
        the model initialization.

        Parameters:
        - data (tuple): A tuple containing the input data and the true labels.

        Returns:
        - dict: A dictionary containing the loss value for the training step.
        """

        x, y_true = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            if self.loss_fun == 'mse':
                loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
            elif self.loss_fun == 'huber':
                loss = tf.keras.losses.Huber()(y_true, y_pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {'loss': loss}

    def test_step(self, data):
        """
        Evaluate the model on the test data.

        This method overrides the test_step method of tf.keras.Model. It computes the loss for the given 
        test data without training the model. The loss function used is specified during the model 
        initialization.

        Parameters:
        - data (tuple): A tuple containing the input data and the true labels.

        Returns:
        - dict: A dictionary containing the loss value for the test step.
        """

        x, y_true = data
        y_pred = self(x, training=False)
        if self.loss_fun == 'mse':
            loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
        elif self.loss_fun == 'huber':
            loss = tf.keras.losses.Huber()(y_true, y_pred)
        return {'loss': loss}