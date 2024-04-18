from scipy.stats import spearmanr
import numpy as np
import pandas as pd
from math import log2
import json

from PIL import Image, ImageDraw, ImageFont

import nltk
from nltk.stem import WordNetLemmatizer

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

class CoocHandler:
    def __init__(self, cooc_file):
        self.nuv = dict()
        self.nu = dict()
        self.n = 0
        with open(cooc_file) as fin:
            for line in fin.readlines():
                modality, first_token, *records = line.strip().split()
                for rec in records:
                    second_token, cooc_val = rec.split(":")

                    if not first_token in self.nuv:
                        self.nuv[first_token] = dict()
                    self.nuv[first_token][second_token] = float(cooc_val)

                    if not second_token in self.nuv:
                        self.nuv[second_token] = dict()
                    self.nuv[second_token][first_token] = float(cooc_val)

                    self.nu[first_token] = self.nu.get(first_token, 0) + float(cooc_val)
                    self.nu[second_token] = self.nu.get(second_token, 0) + float(cooc_val)
                    self.n +=  float(cooc_val)

    def pmi(self, u, v):
        if not u in self.nu or not v in self.nu or not v in self.nuv[u]:
            return 0
        return log2(self.n * self.nuv[u][v] / (self.nu[u] * self.nu[v]))

    def pmi2(self, u, v):
        if not u in self.nu or not v in self.nu or not v in self.nuv[u]:
            return 0
        return log2(self.nuv[u][v] ** 2 / (self.nu[u] * self.nu[v]))

    def pmi3(self, u, v):
        if not u in self.nu or not v in self.nu or not v in self.nuv[u]:
            return 0
        return log2(self.nuv[u][v] ** 3 / (self.n * self.nu[u] * self.nu[v]))

    def ppmi(self, u, v):
        if not u in self.nu or not v in self.nu or not v in self.nuv[u]:
            return 0
        return max(log2(self.n * self.nuv[u][v] / (self.nu[u] * self.nu[v])), 0)

    def npmi(self, u, v):
        if not u in self.nu or not v in self.nu or not v in self.nuv[u]:
            return 0
        return self.pmi(u, v) / log2(self.nuv[u][v] / self.n)


class MetricsStorage:
    def __init__(self):
        self.metrics = dict()

    def add(self, metric_name, metric_values):
        self.metrics[metric_name] = metric_values

    def getCorrelationList(self, metric_values):
        metrics = [metric_name for metric_name in self.metrics]
        stats = [spearmanr(metric_values, self.metrics[metric_name]).statistic for metric_name in self.metrics]
        return (metrics, stats)

    def getCorrelationTable(self, metric_values):
        metrics = [metric_name for metric_name in self.metrics]
        stats = [spearmanr(metric_values, self.metrics[metric_name]).statistic for metric_name in self.metrics]
        stats = list(map(lambda x: round(x, 2), stats))
        return pd.DataFrame(list(zip(metrics, stats)), columns=["metric", "correlation coef"])

    def dumpJson(self, filename):
        with open(filename, "w") as outfile: 
            json.dump({key: val.tolist() for key, val in self.metrics.items()}, outfile)

    def loadJson(self, filename):
        with open(filename) as infile: 
            data = json.load(infile)
            self.metrics = {key: np.array(val) for key, val in data.items()}

def compareTopWordsVsWeithedCoh(raw_text, topic_dist, font_size=18):
    def get_apacity(alpha):
        return "%0.2X" % int(255 * alpha)

    image = Image.new('RGBA', (1000, 400), color = 'white')
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=font_size)
    text_color = 'black'
    wnl = WordNetLemmatizer()

    top_words = topic_dist.iloc[np.argsort(topic_dist.to_numpy())[::-1]][:10]
    x = 10
    y = 10
    for line in raw_text.split('\n'):
        for word in nltk.wordpunct_tokenize(line):
            lem = wnl.lemmatize(word)
            if lem in topic_dist:
                bbox = draw.textbbox((x, y), word, font=font)
                draw.rectangle(bbox,  fill='#00ff00' + get_apacity(topic_dist[lem] / topic_dist.max()))

            if lem in top_words:
                bbox = draw.textbbox((x, y), word, font=font)
                draw.rectangle(bbox,  fill='#ff0000')
            draw.text((x, y), word, fill=text_color, font=font)
            x += draw.textlength(word, font=font) + 5
        x = 0
        y += font_size + 5
        
    return image
