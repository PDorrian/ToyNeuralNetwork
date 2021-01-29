import random
import sys


def preprocess_data(file_name, category):
    try:
        file = open(file_name)
    except:
        print("ERROR: File '" + file_name + "' not found.")
    # Identify attribute to be classified
    head = file.readline()
    head = head.replace("\n", "");
    head = head.split(',')
    try:
        cat_index = head.index(category)
    except ValueError:
        print("ERROR: Category '" + category + "' not found.")
        sys.exit(1)

    lines = file.readlines()
    # Calculate min and max values for normalization
    mins = [1000]*9
    maxs = [0]*9
    classes = set()
    for line in lines:
        line = line.replace("\n", "");
        data = line.split(',')
        classes.add(data[cat_index])
        del data[cat_index]
        data = [float(x) for x in data]

        for i in range(len(data)):
            maxs[i] = max(data[i], maxs[i])
            mins[i] = min(data[i], mins[i])

    # Format data
    classes = list(classes)
    output_size = len(classes)
    unsorted_data = {}
    for line in lines:
        line = line.replace("\n", "");
        data = line.split(',')
        target = data[cat_index]
        del data[cat_index]
        data = [float(x) for x in data]

        # Normalise each attribute to range 0-1
        for i in range(len(data)):
            data[i] = (data[i]-mins[i])/(maxs[i]-mins[i])

        # Convert target output to vector
        output = [0]*output_size
        output[classes.index(target)] = 1

        # Save to dictionary in order
        unsorted_data[tuple(data)] = output

    # Randomly shuffle the keys
    keys = list(unsorted_data.keys())
    random.shuffle(keys)
    # Add two-thirds of the data to the training set
    training_data = {}
    for i in range(2*len(unsorted_data)//3):
        key = keys.pop()
        training_data[key] = unsorted_data[key]
    # Add the remaining data to the testing set
    testing_data = {}
    for i in range(len(keys)):
        key = keys[i]
        testing_data[key] = unsorted_data[key]

    file.close()
    head.remove(category)
    return training_data, testing_data, head, classes

