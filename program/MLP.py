# -*- coding: utf-8 -*-

from typing import List, Any, Union
import numpy as np
from ReadDate import input_data
import matplotlib.pyplot as plt
from pylab import *

def softmax(x):
    x_ = np.sum(np.exp(x), axis=1)
    return np.exp(x) / np.resize(x_, [x_.shape[0], 1])

def tanhx(x):
    return  np.tanh(x)

def tanhx_(x):
    return  (1 - np.multiply((np.tanh( x)), (np.tanh(x))))

# 采用带有动量的随机梯度下降优化算法
def FeedForward(X_train_, Y_train_, X_test, Y_test, hidden, activations, activations_, eta_=0.1, batch_size=128, epochs = 150, holdoutRatio = 0.2,momentum=0):
#采用Adam算法进行优化
#def FeedForward(X_train_, Y_train_, X_test, Y_test, hidden, activations, activations_, eta_=0.1, batch_size=128, epochs=150, holdoutRatio=0.2, alp1=0.9,alp2=0.99,alph=0.00000001):
    #Split Data into Training and Validation
    indxs = np.random.permutation(X_train_.shape[0])
    idxs = indxs[:int((1 - holdoutRatio) * X_train_.shape[0])]
    X_train = []
    Y_train = []
    X_valid = []
    Y_valid = []
    for i in range(X_train_.shape[0]):
        if i in idxs:
            X_train.append(np.resize(X_train_[i, :], (X_train_.shape[1])))
            Y_train.append(np.resize(Y_train_[i, :], (Y_train_.shape[1])))
        else:
            X_valid.append(np.resize(X_train_[i, :], (X_train_.shape[1])))
            Y_valid.append(np.resize(Y_train_[i, :], (Y_train_.shape[1])))
    X_train = np.matrix(X_train)
    Y_train = np.matrix(Y_train)
    X_valid = np.matrix(X_valid)
    Y_valid = np.matrix(Y_valid)
    print('Train-Valid split done.')
    Weights = []
    prev = X_train.shape[1]

    #Initialize Weight
    for i in range(len(hidden)):
        curr = hidden[i]
        tmp = np.random.normal(0, 1.0 / np.sqrt(prev + 1), [prev + 1, curr])
        Weights.append(tmp)
        prev = curr
    Weights.append(np.random.normal(0, 1.0 / np.sqrt(prev + 1), [prev + 1, Y_train.shape[1]]))

    def forwardPass(Xdata, A, Z):
        curr_ = Xdata
        for i in range(len(Weights)):
            curr_ = np.hstack((curr_, np.ones([curr_.shape[0], 1])))
            curr_ = np.dot(curr_, Weights[i])
            A.append(np.hstack((curr_, np.ones([curr_.shape[0], 1]))))
            curr_ = activations[i](curr_)
            Z.append(np.hstack((curr_, np.ones([curr_.shape[0], 1]))))
        return curr_

    def forwardPass_(Xdata, A, Z):
        curr_ = Xdata
        for i in range(len(laWeights)):
            curr_ = np.hstack((curr_, np.ones([curr_.shape[0], 1])))
            curr_ = np.dot(curr_, laWeights[i])
            A.append(np.hstack((curr_, np.ones([curr_.shape[0], 1]))))
            curr_ = activations[i](curr_)
            Z.append(np.hstack((curr_, np.ones([curr_.shape[0], 1]))))
        return curr_

    def calculateLoss(Xtrain, Ytrain, A, Z):
        E_ = 0.0
        curr_ = forwardPass(Xtrain, A, Z)
        for i in range(Ytrain.shape[0]):
            for j in range(Ytrain[0].shape[0]):
                if Ytrain[i, j] == 1.0:
                    E_ -= np.log(curr_[i, j])
        return E_ / len(Xtrain)

    def calculateAccuracy(Xtest, Ytest, A, Z):
        curr_ = forwardPass(Xtest, A, Z)
        curr_ = np.argmax(curr_, axis=1)
        count = 0
        for i in range(len(Ytest)):
            if int(Ytest[i, curr_[i]]) == 1:
                count += 1
        return count * 1.0 / len(Xtest)

    def shuffle(X_tr, Y_tr):
        indxs = np.random.permutation(X_tr.shape[0])
        idxs = indxs[:X_tr.shape[0]]
        X_tr_ = []
        Y_tr_ = []
        for i in idxs:
            X_tr_.append(np.resize(X_tr[i, :], (X_train_.shape[1])))
            Y_tr_.append(np.resize(Y_tr[i, :], (Y_train_.shape[1])))
        X_tr_ = np.matrix(X_tr_)
        Y_tr_ = np.matrix(Y_tr_)
        return X_tr_, Y_tr_

    #Epochs
    prev_mom = {}
    laWeights = []
    s={}
    r={}

    for i in range(len(Weights)):
        s[i] = np.zeros((Weights[i].shape[0], Weights[i].shape[1]))
        r[i]=np.zeros((Weights[i].shape[0], Weights[i].shape[1]))

        prev_mom[i] = np.zeros((Weights[i].shape[0],Weights[i].shape[1]))
        laWeights.append(Weights[i])

    training_loss = []
    validation_loss = []
    training_accuracy: List[Union[float, Any]] = []
    validation_accuracy = []
    saved_weights = []
    print('Initialization Complete. Training Begins...\n')
    t=0#初始化时间步
    for e in range(epochs):
        eta = eta_
        tra_shape = X_train.shape[1]
        temp = np.hstack((X_train, Y_train))
        np.random.shuffle(temp)
        X_tra = temp[:batch_size, :tra_shape]
        Y_tra = temp[:batch_size, tra_shape:]
        Z = []
        A = []
        delta = {}
        curr = forwardPass_(X_tra, A, Z)
        t = t + 1
        delta[len(laWeights) - 1] = Y_tra - curr
        i = len(laWeights) - 1
        #SGD
        while (i >= 0):
            if i == len(laWeights) - 1:
                delta[i - 1] = np.multiply(activations_[i - 1](A[i - 1]), np.dot(delta[i], laWeights[i].T))
            elif i > 0:
                delta[i - 1] = np.multiply(activations_[i - 1](A[i - 1]), np.dot(delta[i][:, :-1], laWeights[i].T))
            if i == 0:
                upd = momentum * prev_mom[i] - eta * np.dot(np.hstack((X_tra, np.ones([X_tra.shape[0], 1]))).T,
                                                            delta[i][:, :-1]) / X_tra.shape[0]
            elif i == len(laWeights) - 1:
                upd = momentum * prev_mom[i] - eta * np.dot(Z[i - 1].T, delta[i]) / X_tra.shape[0]
            else:
                upd = momentum * prev_mom[i] - eta * np.dot(Z[i - 1].T, delta[i][:, :-1]) / X_tra.shape[0]
            Weights[i] = Weights[i] - upd
            prev_mom[i] = upd
            laWeights[i] = Weights[i] - momentum * upd
            i -= 1

        #Adam
        # while (i >= 0):
        #     if i == len(laWeights) - 1:
        #         delta[i - 1] = np.multiply(activations_[i - 1](A[i - 1]), np.dot(delta[i], laWeights[i].T))
        #     elif i > 0:
        #         delta[i - 1] = np.multiply(activations_[i - 1](A[i - 1]), np.dot(delta[i][:, :-1], laWeights[i].T))
        #     if i == 0:
        #         g = np.dot(np.hstack((X_tra, np.ones([X_tra.shape[0], 1]))).T,
        #                                                     delta[i][:, :-1]) / X_tra.shape[0]
        #         s1=alp1*s[i]+(1-alp1)*g
        #         r1=alp2*r[i]+(1-alp2)*(np.multiply(g,g))
        #     elif i == len(laWeights) - 1:
        #         g= np.dot(Z[i - 1].T, delta[i]) / X_tra.shape[0]
        #         s1 = alp1 * s[i] + (1 - alp1) * g
        #         r1 = alp2 * r[i] + (1 - alp2) * (np.multiply(g, g))
        #     else:
        #         g= np.dot(Z[i - 1].T, delta[i][:, :-1]) / X_tra.shape[0]
        #         s1 = alp1 * s[i] + (1 - alp1) * g
        #         r1 = alp2 * r[i] + (1 - alp2) * (np.multiply(g, g))
        #     ss=s1/(1-pow(alp1, t))
        #     rr=r1/(1-pow(alp2, t))
        #     upd1=eta*ss
        #     upd2=np.sqrt(rr)+alph
        #     upd=np.divide(upd1,upd2)
        #     Weights[i] = Weights[i] + upd
        #     laWeights[i] = Weights[i]
        #     i -= 1

        training_accuracy.append(calculateAccuracy(X_train, Y_train, A, Z))
        validation_accuracy.append(calculateAccuracy(X_valid, Y_valid, A, Z))
        training_loss.append(calculateLoss(X_train, Y_train, A, Z))
        validation_loss.append(calculateLoss(X_valid, Y_valid, A, Z))
        saved_weights.append(Weights)
        if e%10 == 0:
            print('Epoch = ', e, 'Train Acc = ', training_accuracy[e], 'Valid Acc = ', validation_accuracy[e],'Train Loss = ', training_loss[e], 'Valid Loss = ', validation_loss[e])

    plt.title('training_accuracy')
    plt.xlabel('number of epoch')
    plt.ylabel('training_accuracy')
    plt.plot(np.linspace(0,epochs-1,epochs),training_accuracy,'b-',label='122')
    plt.show()
    plt.title('validation_accuracy')
    plt.xlabel('number of epoch')
    plt.ylabel('validation_accuracy')
    plt.plot(np.linspace(0, epochs - 1, epochs), validation_accuracy, 'b-', label='122')
    plt.show()
    plt.title('training_loss')
    plt.xlabel('number of epoch')
    plt.ylabel('training_loss')
    plt.plot(np.linspace(0, epochs - 1, epochs), training_loss, 'b-', label='122')
    plt.show()
    plt.title('validation_loss')
    plt.xlabel('number of epoch')
    plt.ylabel('validation_loss')
    plt.plot(np.linspace(0, epochs - 1, epochs), validation_loss, 'b-', label='122')
    plt.show()
    training_accuracy = np.array(training_accuracy)
    validation_accuracy = np.array(validation_accuracy)
    training_loss = np.array(training_loss)
    validation_loss = np.array(validation_loss)

    idx = np.argmax(validation_accuracy)
    final_weights = saved_weights[idx]

    A = []
    Z = []
    Weights = final_weights
    testing_accuracy = calculateAccuracy(X_test, Y_test, A, Z)
    return idx, training_accuracy[idx], validation_accuracy[idx], training_loss[idx], validation_loss[
        idx], testing_accuracy, final_weights


def main():
    print('Data Loaded.')
    train_images, train_labels, test_images, test_labels = input_data()
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    tmp = np.zeros([len(train_labels), 10])
    for i in range(len(train_labels)):
        tmp[i, train_labels[i]] = 1
    labels_train = np.matrix(tmp)
    tmp = np.zeros([len(test_labels), 10])
    for i in range(len(test_labels)):
        tmp[i, test_labels[i]] = 1
    labels_test = np.matrix(tmp)
    print('Labels One-Hot Encoded.')
    ep = 200
    # SDG
    idx, tr_a, va_a, tr_l, va_l, te_acc, final_weights = FeedForward(train_images, labels_train, test_images,
                                                                     labels_test, [100, 50,25], [tanhx, tanhx,tanhx, softmax],
                                                                     [tanhx_, tanhx_,tanhx_], eta_=0.03,batch_size=128, epochs=ep)

    #SDG with momentu
    # idx, tr_a, va_a, tr_l, va_l, te_acc, final_weights = FeedForward(train_images, labels_train, test_images,
    #                                                                  labels_test, [100, 50,25], [tanhx, tanhx,tanhx, softmax],
    #                                                                  [tanhx_, tanhx_,tanhx_], eta_=0.03,batch_size=128, epochs=ep,momentu=0.9)

    #Adam
    # idx, tr_a, va_a, tr_l, va_l, te_acc, final_weights = FeedForward(train_images, labels_train, test_images,
    #                                                                 labels_test, [100, 50,25], [tanhx, tanhx,tanhx, softmax],
    #                                                                  [tanhx_, tanhx_,tanhx_], eta_=0.0015,batch_size=128, epochs=ep,
    #                                                                 alp1=0.9,alp2=0.99,alph=0.0000000)

    print('Best Fit Obtained at Epoch : ', idx)
    print('Training Loss at best fit : ', tr_l)
    print('Validation Loss at best fit : ', va_l)
    print('Training Accuracy at best fit : ', tr_a)
    print('Validation Accuracy at best fit : ', va_a)
    print('Testing Accuracy at best fit : ', te_acc)


if __name__ == "__main__":
    main()