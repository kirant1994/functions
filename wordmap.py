import numpy as np

class WordMap:
    def __init__(self, words):
        self.markers = ['<s>', '<e>']
        self.words = [item.upper() for item in np.unique(words)]
        self.words = ['_'] + self.markers + [' ', '<unk>'] + self.words
        self.int2word = dict(zip(np.arange(0, len(self.words)), self.words))
        self.word2int = dict(zip(self.words, np.arange(0, len(self.words))))

    # If add_spaces is True, a space will be added in between every word
    # If iterable is True, the sentence won't be split by spaces
    def word_to_int(self, sentence, add_spaces=True, iterable=False):
        sentence = sentence.upper()
        if not iterable:
            sentence_split = sentence.split(' ')
        else:
            sentence_split = sentence
        if add_spaces:
            word_list = []
            [word_list.extend([item] + [' ']) for item in sentence_split]
            word_list = word_list[:-1]
        else:
            word_list = sentence_split
        result = []
        for idx, item in enumerate(word_list):
            result.append(self.word2int.get(item, self.word2int['<unk>']))
        result = [self.word2int['<s>']] + result + [self.word2int['<e>']]
        return result

    def int_to_word(self, int_list):
        print(int_list)
        result = [self.int2word[item] for item in int_list]
        return result

    def word_to_int_batched(self, sentences, add_spaces=True, iterable=False):
        outs = [self.word_to_int(item, add_spaces=add_spaces, iterable=iterable) for item in sentences]
        lens = [len(item) for item in outs]
        mx_len = np.max(lens)
        batch = np.zeros((len(sentences), mx_len)) + 2
        for idx, item in enumerate(outs):
            batch[idx, :len(item)] = np.array(item)
        return batch, lens
    
    def make_string(self, word_list):
        temp_str = ''
        for item in word_list:
            if item not in self.markers:
                temp_str += item
            if item  == '<e>':
                break
        return temp_str
    
    def int_to_word_batched(self, int_lists):
        outs = [self.int_to_word(item) for item in int_lists]
        outs = [self.collapse(item) for item in outs]
        outs = [self.make_string(item) for item in outs]
        return outs
    
    def collapse(self, word_list):
        word_list_collapsed = []
        for idx, item in enumerate(word_list[:-1]):
            if word_list[idx] != word_list[idx + 1]:
                word_list_collapsed.append(word_list[idx])
        word_list_collapsed.append(word_list[-1])
        return word_list_collapsed

if __name__ == '__main__':
    sentences = ['My name is Kiran', 'My name is not Sanju', 'Hello Hello World']
    wordmap = WordMap(['My', 'name', 'is', 'Kiran', 'Sanju', 'not', 'Hello', 'World'])
    wordmap = WordMap(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
    result_i, lens = wordmap.word_to_int_batched(sentences, add_spaces=False, iterable=True)
    result_c = wordmap.int_to_word_batched(result_i)
    print(result_i, lens)
    print(result_c)