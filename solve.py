import numpy as np

WLEN = 5   # Length of allowed words
NCHAR = 26 # Number of characters in the alphabet
ORD_A = ord('a') # Value of 'a' in the encoding

def load_words_as_array_of_int(fname):
    """
    Convert a list of words of length WLEN into a numpy array of type byte, with 'a' in the
    original word corresponding to zero in the output.
    """

    wlist_orig = open(fname).read().splitlines()

    wlist = np.zeros((len(wlist_orig), WLEN), dtype = np.byte)
    for i, word in enumerate(wlist_orig):
        for j, char in enumerate(word):
            wlist[i, j] = ord(char) - ORD_A

    return wlist

wlist = load_words_as_array_of_int('wordlist.txt')
solns = load_words_as_array_of_int('solutions.txt')

NW = len(wlist) # Number of words, solutions
NS = len(solns)

letters_present = np.zeros((NS, NCHAR), dtype = 'bool') # For each solution, keep track of characters present
for i, sol in enumerate(solns):
    for lett in sol:
        letters_present[i, lett] = True

def get_num_compatible(wpick, spick):
    """
    Given our first pick and the actual solution, how many solutions get a result
    consistent with the true solution (on the given pick)
    """

    # At the beginning, all solutions are compatible with wpick/spick
    filt = np.ones(NS, dtype = 'bool')
    for i in range(WLEN):
        # If the chosen word agrees with the solution on a given letter,
        # we reduce the allowed solutions to only those where i-th letter
        # matches
        if wpick[i] == spick[i]:
            filt *= (solns[:, i] == spick[i])
        else:
            #Otherwise check for appearance of the given letter in the solution
            num_appear = np.sum(spick == wpick[i])
            if num_appear > 0:
                filt *= letters_present[:, wpick[i]] == 1
            else:
                filt *= letters_present[:, wpick[i]] == 0

    return np.sum(filt)

wpick = wlist[0]
spick = solns[1]
expected_remaining = np.zeros(NW)
remaining = np.zeros(NS, dtype = int)

best = 1000000

for i in range(NW):
    if i % 500 == 0:
        print('epoch', i)
    for j in range(NS):
        remaining[j] = get_num_compatible(wlist[i], solns[j])
    expected_remaining[i] = np.mean(remaining)
    if expected_remaining[i] < best:
        best = expected_remaining[i]
        print(f'{best:.3f}', ''.join([chr(ORD_A + c) for c in wlist[i]]))
