import numpy as np


class MatrixFactorizationModel:
    """
    Basic Matrix Factorization Model
    """

    def __init__(self, x_train, y_train, x_test, y_test, num_users, num_movies, k, alpha, beta, epochs, shuffle=False, verbose=False):
        """

        :param x_train (ndarray) : 훈련 데이터 (userId, movieId)
        :param y_train (ndarray) : 훈련 데이터 라벨 (rating)
        :param x_test (ndarray)  : 테스트 데이터 (userId, movieId)
        :param y_test (ndarray)  : 테스트 데이터 라벨 (rating)
        :param num_users (int)   : 전체 user 수
        :param num_movies (int)  : 전체 movie 수
        :param k (int)           : latent feature 크기
        :param alpha (float)     : learning rate
        :param beta (float)      : lambda, regularization parameter
        :param epochs (int)      : training epochs
        :param shuffle (bool)    : 훈련 데이터 shuffle
        :param verbose (bool)    : print status
        """

        self.train_size = x_train.shape[0]
        self.x_train = x_train
        self.y_train = y_train
        self.test_size = x_test.shape[0]
        self.x_test = x_test
        self.y_test = y_test
        self.num_users = num_users
        self.num_movies = num_movies
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.epochs = epochs
        self.shuffle = shuffle
        self.verbose = verbose

        # latent feature 초기화
        self.P = np.random.normal(size=(self.num_users+1, self.k))   # user latent feature
        self.Q = np.random.normal(size=(self.num_movies+1, self.k))  # movie latent feature

        # bias 초기화
        self.b_u = np.zeros(self.num_users)                          # user bias
        self.b_i = np.zeros(self.num_movies)                         # movie bias
        self.b = np.mean(self.y_train)                               # global bias, Mu, overall average rating

    def train(self):
        """
        Matrix Factorization Model 학습, optimizer로 SGD를 이용하여 가중치 갱신
        :return: training_process (학습 진행 상황, epoch 별 rmse 값)
        """
        training_process = []
        for epoch in range(self.epochs):
            cnt = 0
            if self.shuffle is True:
                idx = np.arange(self.x_train.shape[0])
                np.random.shuffle(idx)
                for (i, j), r in zip(self.x_train[idx], self.y_train[idx]):
                    self.sgd(i, j, r)
                    cnt += 1
                    if self.verbose is True and cnt % 500000 == 0:
                        s = "." * (cnt // 500000)
                        print("\rtraining", s, end='')
            else:
                for (i, j), r in zip(self.x_train, self.y_train):
                    self.sgd(i, j, r)
                    cnt += 1
                    if self.verbose is True and cnt % 500000 == 0:
                        s = "." * (cnt // 500000)
                        print("\rtraining", s, end='')

            rmse = self.rmse()
            _, test_error = self.test()
            training_process.append([epoch+1, rmse, test_error])

            if self.verbose is True:
                print(" Epoch: %d, rmse = %.4f, test_error(rmse) = %.4f" % (epoch + 1, rmse, test_error))

        return training_process

    def test(self):
        """

        :param x_test (ndarray) : test data (userId, movieId)
        :param y_test (ndarray) : test data label(ratings)
        :return: preds, rmse (테스트 데이터에 대한 에측 값, rmse 값)
        """
        preds = []  # test data에 대한 예측 값 리스트
        error = 0
        for (i, j), r in zip(self.x_test, self.y_test):    # i: userId, j: movieId, r: 실제 rating 값
            pred = self.get_pred(i, j)
            preds.append(str(round(pred, 4)))
            error += pow(r - pred, 2)

        return preds, np.sqrt(error / self.test_size)

    def rmse(self):
        """
        train data에 대한 rmse 값 계산
        :return: rooted mean square error 값
        """
        error = 0
        for (i, j), r in zip(self.x_train, self.y_train):    # i: userId, j: movieId, r: 실제 rating 값
            pred = self.get_pred(i, j)
            error += pow(r - pred, 2)
        return np.sqrt(error / self.train_size)

    def sgd(self, i, j, r):
        """
        Stochastic Gradient Descent 수행

        :param i (int)   : userId
        :param j (int)   : movieId
        :param r (float) : 실제 rating 값
        """
        pred = self.get_pred(i, j)
        error = r - pred

        # latent feature 갱신
        self.P[i, :] += self.alpha * (error * self.Q[j, :] - self.beta * self.P[i, :])
        self.Q[j, :] += self.alpha * (error * self.P[i, :] - self.beta * self.Q[j, :])

        # bias 갱신
        self.b_u[i] += self.alpha * (error - self.beta * self.b_u[i])
        self.b_i[j] += self.alpha * (error - self.beta * self.b_i[j])

    def get_pred(self, i, j):
        """

        :param i (int)   : userId
        :param j (int)   : movieId
        :return: user i, movie j에 대해 모델이 예측한 rating 값
        """
        pred = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return pred

    def get_pred_matrix(self):
        """

        :return: 모든 user x movie 조합에 대하여 예측한 matrix
        """
        return self.b + self.b_u[:, np.newaxis] + self.b_i[np.newaxis:, ] + self.P.dot(self.Q.T)
