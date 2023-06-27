import numpy as np

class FiringRate():
    """Base Firing rate model

    Extending subclasses should override the `forward`, `update` and `backward`
    methods to provide new learning rules.
    """
    def __init__(self, input_neurons, output_neurons):
        self.input_neurons = input_neurons
        self.weights = np.random.random((output_neurons, input_neurons))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x):
        return np.dot(self.weights, x.T)
    
    def update(self, w):
        self.weights = w.copy()

    def backward(self):
        pass

class HebbFiringRate(FiringRate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        self.last_x = x
        self.last_y = super().forward(x)
        return self.last_y

    def backward(self, eta = 1e-3):
        w_new = self.weights + eta * self.last_y * self.last_x
        self.update(w_new)

class OjaFiringRate(HebbFiringRate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def backward(self, eta = 1e-3, alpha = 1e-1):
        w_new = self.weights + eta * \
                (self.last_y*self.last_x - alpha*self.last_y**2 * self.weights)
        self.update(w_new)

class SubtractiveNormFiringRate(HebbFiringRate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def backward(self, eta = 1e-3, alpha = 1):
        w_new = self.weights + eta * \
                (self.last_y * self.last_x - \
                (self.last_y * self.last_x.sum()).sum()/self.input_neurons)
        self.update(w_new)

class BCMFiringRate(HebbFiringRate):
    def __init__(self, *args, theta0 = 1e-5, tau = 1e-7):
        super().__init__(*args)
        self.theta = theta0
        self.tau = tau

    def forward(self, *args, **kwargs):
        super().forward(*args, **kwargs)
        self.theta = self.theta + self.tau * ((self.last_y**2).item() - self.theta)

    def backward(self, eta = 1e-3):
        w_new = self.weights + eta * (self.last_y*self.last_x*(self.last_y - self.theta))
        self.update(w_new)


