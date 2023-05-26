class Patient_stats:

    def __init__(self):
        self.worst_idx = -1
        self.best_idx = -1
        self.worst_val_acc = 100
        self.best_val_acc = 0
        self.cv_confusion_matrix = None


    def set(self, worst_idx, best_idx, worst_val_acc, best_val_acc):
        self.worst_idx = worst_idx
        self.best_idx = best_idx
        self.worst_val_acc = worst_val_acc
        self.best_val_acc = best_val_acc


    def set_best(self, best_idx, best_val_acc):
        self.best_idx = best_idx
        self.best_val_acc = best_val_acc


    def set_worst(self, worst_idx, worst_val_acc):
        self.worst_idx = worst_idx
        self.worst_val_acc = worst_val_acc

    def print_stats(self):
        print("//  Best predicted patient: " + str(self.best_idx) + ", VAL. ACC.: " + str(self.best_val_acc) + "//")
        print("// Worst predicted patient: " + str(self.worst_idx) + ", VAL. ACC.: " + str(self.worst_val_acc) + "//")