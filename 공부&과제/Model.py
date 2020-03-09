import numpy as np
import math

class TwoLayerNet():


    def __init__(self, X, input_size, hidden_size, output_size, std=1e-4):

        self.params = {}
        self.params["W1"] = std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.random.randn(hidden_size)
        self.params["W2"] = std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.random.randn(output_size)

    def forward(self, X, y=None):


        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        N, D = X.shape

        # 여기에 p를 구하는 작업을 수행하세요.

        h = np.dot(X, W1) + b1   # 1 layer
        a = np.maximum(0, h)    # activation relu  : 입력된 값과 0을 비교해서 더 큰값 반환 
        o = np.dot(a, W2) + b2  # 2 layer
        p = np.exp(o)/np.sum(np.exp(o),axis=1).reshape(-1,1)   # softmax 확률계산 p =(N,C)

        if y is None:
            return p, a
        
        # 여기에 Loss를 구하는 작업을 수행하세요.
        # Error = -log(정답 label의 Softmax probability)의 총 합계
        # log내에 들어가는 확률값이 1에 가까울수록 -log변환의 값은 작다.
        
        # https://deepnotes.io/softmax-crossentropy 
        # H(y,p)=−∑iyilog(pi)
        
        # N개의 자료에 대한 오차의 합계
        log_likelihood=0
        log_likelihood -= np.log(p[np.arange(N), y]).sum()
        Loss = log_likelihood / N
        
        print('loss : ',Loss)

        return Loss



    def backward(self, X, y, learning_rate=0.00001):

        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        N = X.shape[0] # 데이터 개수
        grads = {}

        
        # p = softmax 확률 결과값
        # a = activation 한 값
        p, a = self.forward(X)

        # 여기에 파라미터에 대한 미분을 저장하세요.
        
        # p의 미분값을 구하기 위한 dp
        dp = p
        for i in range(p.shape[0]):  # 행  N
            for j in range(p.shape[1]):  # 열 C
                if(j==y[i]):   
                    dp[i][j]-=1  # 정답 레이블에 해당하는 노드의 그래디언트는 확률값에서 1을 빼준 값        
          # dL/dS = p-y
        
        # dL/dA = dL/dS * W.T
        da = np.dot(dp, W2.T)
    
        # relu 미분함수 = dA/dH
        dr = np.heaviside(a,0)
        
        # dL/dH = dA/dH * dL/dA
        dh = dr*da

        # 
        grads["W2"] = np.dot(a.T,dp)
        grads["b2"] = np.sum(dp,axis=0)
        grads["W1"] = np.dot(X.T, dh )
        grads["b1"] = np.sum(dh, axis=0)

        
        #가중치 갱신 
        self.params["W2"] -= learning_rate * grads["W2"]
        self.params["b2"] -= learning_rate * grads["b2"]
        self.params["W1"] -= learning_rate * grads["W1"]
        self.params["b1"] -= learning_rate * grads["b1"]

    def accuracy(self, X, y):

        p, _ = self.forward(X)
        
        
        pre_p = np.argmax(p,axis=1)

        return np.sum(pre_p==y)/pre_p.shape[0]