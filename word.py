import math
import csv
class Word:
    unknown_bank = "???????????????????????????????????????????????????????????????????????????????????????????????"
    def __init__(self, length):
        self.letters = list(Word.unknown_bank[0:length])
    def has_letter_been_assigned(self, index):
        if self.letters[index] != "?":
            return True
        return False

def open_dictionary(frequency, rank):
    rank_count = 0
    with open('unigram_freq.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            word, frequency_ = row
            frequency[word]=frequency_
            rank[word] = rank_count
            rank_count = rank_count + 1

def normailze_to_frequency(rank):
    global fequency_lookup
    global rank_lookup
    highest_value = 3.3 * 10**5
    highest_log = math.log2(highest_value)
    normal_value = math.log2(rank)/highest_log
    return normal_value