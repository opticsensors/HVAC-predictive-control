import tensorflow as tf

class ANFIS(tf.keras.Model):
    def __init__(self, 
                 n_inputs, 
                 n_rules, 
                 learning_rate=1e-2,
                 mf='gaussmf',
                 defuzz_method='proportional',
                 loss_fun='mse'
                 ):
        super(ANFIS, self).__init__()
        self.n = n_inputs
        self.m = n_rules
        self.define_variables(mf, defuzz_method)

        self.mf = mf
        self.defuzz_method = defuzz_method
        self.loss_fun = loss_fun
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def define_variables(self, mf, defuzz_method):
        if mf == 'gaussmf':
            self.mu = tf.Variable(tf.random.normal([self.m * self.n]), name="mu")
            self.sigma = tf.Variable(tf.random.normal([self.m * self.n]), name="sigma")

        elif mf == 'gbellmf':
            self.a = tf.Variable(tf.random.normal([self.m * self.n]), name="a")
            self.b = tf.Variable(tf.random.normal([self.m * self.n]), name="b")
            self.c = tf.Variable(tf.random.normal([self.m * self.n]), name="c")

        if defuzz_method == 'proportional':
            self.y = tf.Variable(tf.random.normal([1, self.m]), name="y")

        elif defuzz_method == 'linear':
            #self.coefficients = tf.Variable(tf.random.normal([self.m, self.n]), name="coefficients")
            #self.intercepts = tf.Variable(tf.random.normal([self.n]), name="intercepts") 
            self.coefficients = tf.Variable(tf.random.normal([self.m, self.m]), name="coefficients")
            self.intercepts = tf.Variable(tf.random.normal([self.m]), name="intercepts") 
            
    def call(self, inputs):
        
        # Layer 1: membership layer
        if self.mf == 'gaussmf':
            membership_values = tf.exp(-0.5 * tf.square(tf.subtract(tf.tile(inputs, (1, self.m)), self.mu)) / tf.square(self.sigma))

        elif self.mf == 'gbellmf':
            abs_val = tf.abs((tf.subtract(tf.tile(inputs, (1, self.m)), self.c)) / self.a)
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
        x, y_true = data
        y_pred = self(x, training=False)
        if self.loss_fun == 'mse':
            loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
        elif self.loss_fun == 'huber':
            loss = tf.keras.losses.Huber()(y_true, y_pred)
        return {'loss': loss}