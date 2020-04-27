# I should stop using it, but I can't

import numpy as np

class OldInfoRecorder:

    def __init__(self):
        self.train_smooth_per_test = []
        self.test_smooth_per_test = []
        self.train_acc_per_result = []
        self.test_acc_per_result = []
        self.train_acc_orig = []
        self.test_acc_orig = []
        self.cuts = []
        self.errors = []
        self.volume_errors = []
        self.balance = []

    def record_origin(self, infos_origin):
        train_acc, test_acc = infos_origin
        self.train_acc_orig.append(train_acc)
        self.test_acc_orig.append(test_acc)

    def record_infos_smooth(self, train_smooth, test_smooth):
        self.train_smooth_per_test.append(train_smooth.item())
        self.test_smooth_per_test.append(test_smooth.item())

    def record_info_results(self, infos_results):
        train_acc, test_acc = infos_results
        self.train_acc_per_result.append(train_acc)
        self.test_acc_per_result.append(test_acc)

    def record_cut(self, cut):
        self.cuts.append(cut)

    def record_error(self, error):
        self.errors.append(error.item())

    def record_volume_error(self, error):
        self.volume_errors.append(error)

    def record_balance(self, balance):
        self.balance.append(balance)

    def get_last_train_test_acc(self):
        return self.train_acc_per_result[-1], self.test_acc_per_result[-1]

    def get_avg_train_test_acc(self):
        train_avg = float(np.mean(self.train_acc_per_result))
        test_avg = float(np.mean(self.test_acc_per_result))
        return train_avg, test_avg

    def get_last_train_test_origin(self):
        return self.train_acc_orig[-1], self.test_acc_orig[-1]

    def get_train_test_origin(self):
        train_avg = float(np.mean(self.train_acc_orig))
        test_avg = float(np.mean(self.test_acc_orig))
        return train_avg, test_avg

    def get_cut(self):
        return float(np.mean(self.cuts))

    def get_error(self):
        return float(np.mean(self.errors))

    def get_avg_volume_errors(self):
        return float(np.mean(self.volume_errors))

    def get_last_volume_error(self):
        return float(self.volume_errors[-1])

    def get_last_balance(self):
        return float(self.balance[-1])

    def get_dict(self):
        return {"train_smooth": self.train_smooth_per_test,
                "test_smooth": self.test_smooth_per_test,
                "train_acc": self.train_acc_per_result,
                "test_acc": self.test_acc_per_result,
                "train_acc_orig": self.train_acc_orig,
                "test_acc_orig": self.test_acc_orig,
                "volume_error": self.volume_errors,
                "balance": self.balance}