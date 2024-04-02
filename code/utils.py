from scipy.stats import spearmanr

class PPMIHandler:
    def __init__(self, ppmi_file):
        self.ppmi = dict()
        with open(ppmi_file) as fin:
            for line in fin.readlines():
                modality, first_token, *records = line.strip().split()
                self.ppmi[first_token] = dict()
                for rec in records:
                    second_token, ppmi_val = rec.split(":")
                    self.ppmi[first_token][second_token] = float(ppmi_val)


class MetricsStorage:
    def __init__(self):
        self.metrics = dict()

    def add(self, metric_name, metric_values):
        self.metrics[metric_name] = metric_values

    def getCorrelationList(self, metric_values):
        metrics = [metric_name for metric_name in self.metrics]
        stats = [spearmanr(metric_values, self.metrics[metric_name]).statistic for metric_name in self.metrics]
        return (metrics, stats)