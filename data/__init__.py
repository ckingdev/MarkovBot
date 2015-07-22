def load_sample_data():
    with open("data/sample_data.txt") as f:
        return [line.strip().lower() for line in f.readlines()]
