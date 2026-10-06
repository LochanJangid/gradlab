class LinearRegression:
    def __init__(self):
        self.model = None

    def fit(self, x, y, itr=200, lr=0.03):
        # do every process of training to get min. mse
        self.model = MLP(len(x[0]), [len(y[0])])
        self.label_cnt = len(y[0])
        # evaulate model 
        for yidx in range(self.label_cnt):
            for i in range(itr):
                y_pred = [self.model(xs) for xs in x]
                # calculate mse
                loss = sum((yp[yidx] - yi[yidx])**2 for yp, yi in zip(y_pred, y)) * (1.0 / len(y))
    
                # reset gradients
                for p in self.model.parameters():
                    p.grad = 0.0
    
                # backpropogation
                loss.backward()
    
                # update w & b
                for p in self.model.parameters():
                    p.data = p.data - lr * p.grad

    def predict(self, x):
        # predict the values as learned weights & bias
        return [self.model(xs) for xs in x]