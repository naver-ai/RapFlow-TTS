import os

def MakeDir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text