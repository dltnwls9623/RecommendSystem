import csv
import time
import numpy as np


class DataSet:
    def __init__(self):
        self.userid_to_idx = dict()      # userId를 0~num_user 값으로 변환하기 위한 딕셔너리
        self.movieid_to_idx = dict()     # movieId를 0~num_movie 값으로 변환하기 위한 딕셔너리
        self.x_test = None               # csv 파일 저장을 위한 데이터 (userId, movieId) str 형태로 저장
        self.test_timestamp = None       # csv 파일 저장을 위한 데이터 timestamp str 형태로 저장

    def get_data1(self):
        """
        Dataset1 데이터 로드
        :return: x_train, y_train, x_test, y_test, num_users, num_movies
        """
        TRAIN_TIMESTAMP_START = 1104505203
        TRAIN_TIMESTAMP_END = 1230735592
        TEST_TIMESTAMP_START = 1230735600
        TEST_TIMESTAMP_END = 1262271552
        NUM_TRAIN = 5187587
        NUM_TEST = 930093

        x_train = np.zeros((NUM_TRAIN, 2), dtype=int)  # 훈련 데이터, 형태: (userId, movieId)
        y_train = np.zeros(NUM_TRAIN, dtype=np.float32) # rating

        x_test_id_change = np.zeros((NUM_TEST, 2), dtype=int)       # 테스트 데이터, 형태: (userId, movieId)
        y_test = np.zeros(NUM_TEST, dtype=np.float32) # rating
        self.x_test = []
        self.test_timestamp = []

        print("Loading data...")
        start = time.time()
        with open('../dataset1.csv') as f:
            rdr = csv.reader(f)
            next(rdr)
            user_cnt = 0
            movie_cnt = 0
            train_idx = 0
            test_idx = 0
            for line in rdr:
                timestamp = int(line[3])
                if TRAIN_TIMESTAMP_START <= timestamp <= TRAIN_TIMESTAMP_END:
                    if line[0] not in self.userid_to_idx:     # userId -> idx 딕셔너리에 추가
                        self.userid_to_idx[line[0]] = user_cnt
                        user_cnt += 1
                    if line[1] not in self.movieid_to_idx:    # moiveId -> idx 딕셔너리에 추가
                        self.movieid_to_idx[line[1]] = movie_cnt
                        movie_cnt += 1

                    x_train[train_idx] = np.array([self.userid_to_idx[line[0]], self.movieid_to_idx[line[1]]])
                    y_train[train_idx] = float(line[2])
                    train_idx += 1

                elif TEST_TIMESTAMP_START <= timestamp <= TEST_TIMESTAMP_END:
                    if line[0] not in self.userid_to_idx:     # userId -> idx 딕셔너리에 추가
                        self.userid_to_idx[line[0]] = user_cnt
                        user_cnt += 1
                    if line[1] not in self.movieid_to_idx:   # moiveId -> idx 딕셔너리에 추가
                        self.movieid_to_idx[line[1]] = movie_cnt
                        movie_cnt += 1

                    x_test_id_change[test_idx] = np.array([self.userid_to_idx[line[0]], self.movieid_to_idx[line[1]]])
                    y_test[test_idx] = float(line[2])
                    self.x_test.append(line[:2])
                    self.test_timestamp.append([line[3]])
                    test_idx += 1
        del f

        end = time.time()
        duration = end - start
        mm = duration // 60
        ss = duration % 60
        print("Data loaded in %dmin %.4fsecs!!" % (mm, ss))

        num_users = len(self.userid_to_idx)
        num_movies = len(self.movieid_to_idx)
        print("==== Data Specification ====")
        print("Train Data #: %d" % len(x_train))
        print("Test Data #: %d" % len(self.x_test))
        print("Users #: %d" % num_users)
        print("Movies #: %d" % num_movies)
        print("=============================")

        return x_train, y_train, x_test_id_change, y_test, num_users, num_movies

    def get_data2(self):
        """
        Dataset2 데이터 로드
        :return: x_train, y_train, x_test, y_test, num_users, num_movies
        """
        TIMESTAMP = 1388502017

        print("Loading data...")
        start = time.time()
        num_train = 0
        data = []
        with open('../ratings.csv') as f:
            f.readline()
            while True:
                line = f.readline()
                if not line:
                    break
                line = line[:-1]
                data.append(line)
                d = line.split(',')
                if int(d[3]) < TIMESTAMP:
                    num_train += 1
        del f

        total = len(data)
        num_test = total - num_train

        x_train = np.zeros((num_train, 2), dtype=int)
        y_train = np.zeros(num_train, dtype=np.float32)

        x_test_id_change = np.zeros((num_test, 2), dtype=int)  # 테스트 데이터, 형태: (userId, movieId)
        y_test = np.zeros(num_test, dtype=np.float32)
        self.x_test = []
        self.test_timestamp = []

        train_idx = 0
        test_idx = 0
        user_cnt = 0
        movie_cnt = 0

        for line in data:
            d = line.split(',')
            u = int(d[0])       # userId
            m = int(d[1])       # movieId
            r = float(d[2])     # rating
            t = int(d[3])       # timestamp

            if t < TIMESTAMP:                       # train data
                if u not in self.userid_to_idx:     # userId -> idx 딕셔너리에 추가
                    self.userid_to_idx[u] = user_cnt
                    user_cnt += 1
                if m not in self.movieid_to_idx:    # moiveId -> idx 딕셔너리에 추가
                    self.movieid_to_idx[m] = movie_cnt
                    movie_cnt += 1
                x_train[train_idx] = np.array([self.userid_to_idx[u], self.movieid_to_idx[m]])
                y_train[train_idx] = r
                train_idx += 1
            else:                                   # test_data
                if u not in self.userid_to_idx:     # userId -> idx 딕셔너리에 추가
                    self.userid_to_idx[u] = user_cnt
                    user_cnt += 1
                if m not in self.movieid_to_idx:    # moiveId -> idx 딕셔너리에 추가
                    self.movieid_to_idx[m] = movie_cnt
                    movie_cnt += 1

                x_test_id_change[test_idx] = np.array([self.userid_to_idx[u], self.movieid_to_idx[m]])
                y_test[test_idx] = r
                self.x_test.append(d[:2])
                self.test_timestamp.append(d[3])
                test_idx +=1

        del data
        end = time.time()
        duration = end - start
        mm = duration // 60
        ss = duration % 60
        print("Data loaded in %dmin %.4fsecs!!" % (mm, ss))

        num_users = len(self.userid_to_idx)
        num_movies = len(self.movieid_to_idx)
        print("==== Data Specification ====")
        print("Train Data #: %d" % len(x_train))
        print("Test Data #: %d" % len(self.x_test))
        print("Users #: %d" % num_users)
        print("Movies #: %d" % num_movies)
        print("=============================")

        return x_train, y_train, x_test_id_change, y_test, num_users, num_movies

    def save_result(self, preds, filename='B_results.csv'):
        filename += '.csv'
        csvfile = open(filename, "w", newline="")
        writer = csv.writer(csvfile)

        self.x_test = np.array(self.x_test)
        preds = np.array(preds)
        preds = preds.reshape((-1, 1))
        self.test_timestamp = np.array(self.test_timestamp)
        self.test_timestamp = self.test_timestamp.reshape((-1, 1))
        data = np.concatenate((self.x_test, preds, self.test_timestamp), axis=1)

        for line in data:
            writer.writerow(line)

        csvfile.close()
        print("Result saved!(%s)" % filename)
