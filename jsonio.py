"""
Read raw datasets from file.

"""
import json


def read_infile_as_json(infile):
    """Read infile. the format is set by Quora.

    Arguments:
    ----------
    infile: contains features and labels for the trainning set, and the features
    only for the test set.
    """
    f = file(infile, 'r')
    content = f.readlines()
    # The first line contains the number of training samples.
    train_num = int(content[0])
    data_json = map(json.loads, content[1:])
    # There's one more number separating training and test set, which is not
    # necessary, that's why we use [(train_num + 1):] for the test set.
    return (data_json[0:train_num], data_json[(train_num + 1):])


def read_outfile_as_json(outfile):
    """Read outfile. the format is set by Quora.

    Arguments:
    ----------
    outfile: contains the key and the labels for the test set.
    """
    f = file(outfile, 'r')
    content = f.readlines()
    data_json = map(json.loads, content)
    return data_json
