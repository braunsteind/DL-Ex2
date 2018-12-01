
#this file contains helper functions to deal with the training set and the test set

#let's define some global list:
Representation_Of_Indexes_By_Words = {}
Representation_Of_Words_By_Indexes = {}
Representation_Of_classes_By_Indexes ={}
Representation_Of_Indexes_By_classes ={}
#define set let us assign an index to each unique word - avoid giving the same word many indexes
Dictionary_of_classes = set()
Dictionary_of_words = set()

#more to define:
UNK = "UNKNOWN_WORD"
WINDOW_START = "START"
WINDOW_END = "END"
NEW_LINE = "\n"
TAB = "\t"

def reading_test_dataset(dataset_to_read):
    """

    :param dataset_to_read:
    :return: word_sequences
    """
    #define an empty array to return
    word_sequences = []
    with open(dataset_to_read) as dataset:
        sequence = []
        lines = dataset.readlines()
        for i in lines:
            if NEW_LINE == i:
                word_sequences.append(sequence)
                #clean buffer
                sequence = []
                #continue yo next iteration of the loop
                continue
            #remove new line from string
            word_in_sequence = i.strip(NEW_LINE)
            #all leading and trailing whitespaces are removed from the string.
            word_in_sequence = word_in_sequence.strip()
            sequence.append(word_in_sequence)
    return word_sequences


def divide_word_sequence_into_windows(word_sequences):
    """
    for test set which not has the class for each word
    :param word_sequences:
    :return: return an array of windows when each window contain 5 words (as requiredin the assigment)
    """

    windows_array = []
    for sequence in word_sequences:
        words = []
        #adds start*2, end*2 for each sentence for appropriate windows
        words.append(WINDOW_START)
        words.append(WINDOW_START)
        words.extend(sequence)
        words.append(WINDOW_END)
        words.append(WINDOW_END)

        for index, (cure_word) in enumerate(words):
            if cure_word!=WINDOW_END and cure_word!=WINDOW_START:
                new_window = []
                new_window.append(convert_word_to_index(words[index-2]))
                new_window.append(convert_word_to_index(words[index - 1]))
                new_window.append(convert_word_to_index(cure_word))
                new_window.append(convert_word_to_index(words[index + 1]))
                new_window.append(convert_word_to_index(words[index + 2]))
                windows_array.append(new_window)
    return windows_array

def divide_word_class_sequence_into_windows(word_sequences):
    """
    for test set which not has the class for each word
    :param word_sequences:
    :return: return an array of windows when each window contain 5 words (as requiredin the assigment)
    """

    windows_array = []
    classes = []
    for sequence in word_sequences:
        words_classes_matrix =[]
        words_classes_matrix.extend([(WINDOW_START, WINDOW_START), (WINDOW_START, WINDOW_START)])
        words_classes_matrix.extend(sequence)
        words_classes_matrix.extend([(WINDOW_END, WINDOW_END) , (WINDOW_END, WINDOW_END)])
        #adds start*2, end*2 for each sentence for appropriate windows


        for index, (cure_word,match_class) in enumerate(words_classes_matrix):
            if cure_word!=WINDOW_END and cure_word!=WINDOW_START:
                new_window = []
                new_window.append(convert_word_to_index(words_classes_matrix[index-2][0]))
                new_window.append(convert_word_to_index(words_classes_matrix[index - 1][0]))
                new_window.append(convert_word_to_index(cure_word))
                new_window.append(convert_word_to_index(words_classes_matrix[index + 1][0]))
                new_window.append(convert_word_to_index(words_classes_matrix[index + 2][0]))
                #asign indexes to lists
                classes.append(Representation_Of_classes_By_Indexes[match_class])
                windows_array.append(new_window)
    return windows_array, classes


def reading_train_dataset(train_set):
    """
    :param train_set:
    :return:word_sequences with their classes
    """
    #We use global keyword to read and write a global variable inside a function.
    global Dictionary_of_classes
    global Dictionary_of_words
    # define an empty array to return
    classes_word_sequences = []

    with open(train_set) as dataset:
        sequence = []
        lines = dataset.readlines()
        for i in lines:
            if NEW_LINE == i:
                classes_word_sequences.append(sequence)
                # clean buffer
                sequence = []
                # continue yo next iteration of the loop
                continue
            # remove new line from string
            # all leading and trailing whitespaces are removed from the string.
            i = i.strip(NEW_LINE).strip().strip(TAB)
            word_sequence,match_class = i.split()
            Dictionary_of_words.add(word_sequence)
            Dictionary_of_classes.add(match_class)
            sequence.append((word_sequence, match_class))
    #add the Unknown word to dictionary
    Dictionary_of_words.add(UNK)
    Dictionary_of_classes.add(UNK)

    return classes_word_sequences


def reading_dev_dataset(dev_set):
    """
    :param dev_set:
    :return:word_sequences with their classes
    """
    #We use global keyword to read and write a global variable inside a function.
    global Dictionary_of_classes
    global Dictionary_of_words
    # define an empty array to return
    classes_word_sequences = []

    with open(dev_set) as dataset:
        sequence = []
        lines = dataset.readlines()
        for i in lines:
            if NEW_LINE == i:
                classes_word_sequences.append(sequence)
                # clean buffer
                sequence = []
                # continue yo next iteration of the loop
                continue
            # remove new line from string
            # all leading and trailing whitespaces are removed from the string.
            i = i.strip(NEW_LINE).strip().strip(TAB)
            word_sequence,match_class = i.split()
            sequence.append((word_sequence, match_class))
    return classes_word_sequences


def bring_train_data(train_data):
    """

    :param train_data:
    :return:
    """
    global Representation_Of_Indexes_By_classes
    global Representation_Of_Words_By_Indexes
    global Dictionary_of_classes
    global Representation_Of_classes_By_Indexes
    global Dictionary_of_words
    global Representation_Of_Indexes_By_Words

    #add the first words in our dictionary
    #settings all of the dictionary
    Dictionary_of_words.update(set([WINDOW_START,WINDOW_END]))
    classes_word_sequences = reading_train_dataset(train_data)
    Representation_Of_classes_By_Indexes = {
        m_class: m_index for m_index, m_class in enumerate(Dictionary_of_classes)
    }

    Representation_Of_Indexes_By_Words = {
        m_index: m_word for m_word, m_index in Representation_Of_Words_By_Indexes.iteritems()

    }

    Representation_Of_Indexes_By_classes = {
        m_index: m_class for m_class, m_index in Representation_Of_classes_By_Indexes.iteritems()
    }

    Representation_Of_Words_By_Indexes = {m_word: m_index for m_index, m_word in enumerate(Dictionary_of_words)}

    return divide_word_class_sequence_into_windows(classes_word_sequences)

def bring_test_data(test_data_set):
    word_sequences = reading_test_dataset(test_data_set)
    return divide_word_sequence_into_windows(word_sequences)



def bring_dev_data(dev_data):
    """

    :param dev_data:
    :return: windows of 5 words each and their classes
    """

    classes_word_sequences = reading_dev_dataset(dev_data)
    return divide_word_class_sequence_into_windows(classes_word_sequences)



def convert_word_to_index(word_to_convert):
    """
    The first step in using an embedding layer is to encode this sentence by indices.
     In this case we assign an index to each unique word

    :param word_to_convert:
    :return: return the index which represent the word.
    """

    if word_to_convert not in Representation_Of_Words_By_Indexes:
        #if the word is not in the list its index is UNK - Unknown
        #words we have not seen in the training action we'll map them to the same vector in
        #the embedding matrix - unk
        return Representation_Of_Words_By_Indexes[UNK]
    else:
        #if the word is in the list, return its index
        return Representation_Of_Words_By_Indexes[word_to_convert]

