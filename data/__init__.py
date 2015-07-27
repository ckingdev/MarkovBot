import itertools

def load_sample_data():
    with open("data/sample_data.txt") as f:
        return [line.strip().lower() for line in f.readlines()]

def load_word_list():
    with open("data/wordlist.txt") as f:
        gen = (line for line in f)
        word_list = set(line.strip() for line in itertools.dropwhile(lambda x: x != "---\n", gen))
    return word_list