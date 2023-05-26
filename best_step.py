class BestStep:

    def __init__(self, step_no, minibatch_train_acc, minibatch_loss, val_acc, val_f1, whole_train_acc):
        self.step_no = step_no
        self.minibatch_train_acc = minibatch_train_acc
        self.minibatch_loss = minibatch_loss
        self.val_acc = val_acc
        self.val_f1 = val_f1
        self.whole_train_acc = whole_train_acc
        self.confusion_matrix = None


    def set(self, step_no, minibatch_train_acc, minibatch_loss, val_acc, val_f1, confusion_matrix):
        self.step_no = step_no
        self.minibatch_train_acc = minibatch_train_acc
        self.minibatch_loss = minibatch_loss
        self.val_acc = val_acc
        self.val_f1 = val_f1
        self.confusion_matrix = confusion_matrix

    def set_whole_train_acc(self, whole_train_acc):
        self.whole_train_acc = whole_train_acc

    def set_whole_test_predictions(self, whole_test_predictions):
        self.whole_test_predictions = whole_test_predictions
