import numpy as np

class Alphabet:
    markers = ['<s>', '<e>']
    char_list = ['_'] + markers + [' '] + [chr(item) for item in range(ord('A'), ord('A')+26)]
    int2char = dict(zip(np.arange(0, len(char_list)), char_list))
    char2int = dict(zip(char_list, np.arange(0, len(char_list))))

    @staticmethod
    def char_to_int(sentence):
        char_list = []
        sentence = sentence.upper()
        for idx, item in enumerate(sentence[:-1]):
            sentence = sentence.upper()
            if item not in Alphabet.char_list:
                continue
            char_list.append(item)
            if sentence[idx+1] == sentence[idx]:
                char_list.append('_')
        char_list.append(sentence[-1])
        result = [Alphabet.char2int['<s>']] + [Alphabet.char2int[item] for item in char_list] + [Alphabet.char2int['<e>']]
        return result

    @staticmethod
    def int_to_char(int_list):
        result = [Alphabet.int2char[item] for item in int_list]
        return result

    @staticmethod
    def char_to_int_batched(sentences):
        outs = [Alphabet.char_to_int(item) for item in sentences]
        lens = [len(item) for item in outs]
        mx_len = np.max(lens)
        batch = np.zeros((len(sentences), mx_len)) + 2
        for idx, item in enumerate(outs):
            batch[idx, :len(item)] = np.array(item)
        return batch, lens
    
    @staticmethod
    def make_string(char_list):
        temp_str = ''
        for item in char_list:
            if item not in Alphabet.markers:
                temp_str += item
            if item  == '<e>':
                break
        return temp_str
    
    @staticmethod
    def int_to_char_batched(int_lists):
        outs = [Alphabet.int_to_char(item) for item in int_lists]
        outs = [Alphabet.make_string(item) for item in outs]
        outs = [Alphabet.collapse(item) for item in outs]
        return outs
    
    @staticmethod
    def collapse(sentence):
        sentence_collapsed = ''
        for idx, item in enumerate(sentence[:-1]):
            if sentence[idx] != sentence[idx + 1]:
                sentence_collapsed += sentence[idx]
        sentence_collapsed += sentence[-1]
        return sentence_collapsed

if __name__ == '__main__':
    sentences = ['My name is Kiran', 'My name is not Sanju', 'Hello World']
    result_i, lens = Alphabet.char_to_int_batched(sentences)
    result_c = Alphabet.int_to_char_batched(result_i)
    print(result_i, lens)
    print(result_c)