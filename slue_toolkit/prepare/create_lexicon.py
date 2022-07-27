import fire


def get_valid_char(input_file):
    valid_char = "ABCDEFGHIJKLMNOPQRSTUVWXYZ'".lower()
    if "ner" in input_file:
        spl_char_lst = ["!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "~", "`", "{", "}", "[", "<", ">", "?", "]"]
        valid_char += "".join(spl_char_lst)
    return valid_char

def is_valid(w, valid_char):
    for c in w:
        if c not in valid_char:
            return False
    return True


def main(input_file, output_file):
    valid_char = get_valid_char(input_file)
    print(len(valid_char))

    words = set()
    with open(input_file) as f:
        for line in f:
            for word in line.split():
                words.add(word.lower())

    words = [w for w in words if is_valid(w, valid_char)]

    with open(output_file, "w") as f:
        for word in sorted(words):
            s = word + "\t"
            for c in word:
                s += f"{c} "
            s += "|"
            print(s, file=f)


if __name__ == "__main__":
    fire.Fire(main)
