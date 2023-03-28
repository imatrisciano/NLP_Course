import random

def read_dataset(input_file):
    examples = []
    labels = []
    text_to_id_map = {}

    print(f"Reading dataset '{input_file}'...")

    # open file
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        # read and split lines
        contents = f.read()
        file_as_list = contents.splitlines()

        # shuffle dataset entries
        random.shuffle(file_as_list)

        for line in file_as_list:
            if line.startswith("id"):
                continue

            # split by TAB
            split = line.split("\t")
            text = split[1]
            label = split[2]

            text_to_id_map[text] = split[0]
            examples.append(text)
            labels.append(label)
        f.close()
    return examples, labels, text_to_id_map