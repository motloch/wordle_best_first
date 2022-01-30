"""
Find the optimal starting guess for Wordle, by some metric (see below). Takes about 25
minutes on an old Macbook Pro.

We know the lists of allowed words (wordlist.txt) and solutions (solutions.txt). Then by
making a guess we restrict the potential solutions to a subset. We want to choose a
starting guess for which the expected size of this subset is the smallest possible.
"""

import numpy as np

WLEN = 5         # Word length
NCHAR = 26       # Number of characters in the alphabet
ORD_A = ord('a') # Value of 'a' in the encoding

def load_words_as_array_of_int(fname):
    """
    Load a list of words - each of length WLEN - from a file into a numpy array, one word
    per row. To speed up, use np.byte data type. In the output array, 'a' is represented
    as zero, 'b' as one, ...
    """

    wlist_orig = open(fname).read().splitlines()

    wlist = np.zeros((len(wlist_orig), WLEN), dtype = np.byte)
    for i, word in enumerate(wlist_orig):
        for j, char in enumerate(word):
            wlist[i, j] = ord(char) - ORD_A

    return wlist

# Load dictionaries and get their lengths
wlist = load_words_as_array_of_int('wordlist.txt')
solns = load_words_as_array_of_int('solutions.txt')
NW = len(wlist)
NS = len(solns)

# For speedup, for each possible solution we precompute characters present in it (one hot
# encoding)
letters_present = np.zeros((NS, NCHAR), dtype = 'bool')
for i, sol in enumerate(solns):
    for lett in sol:
        letters_present[i, lett] = True

def get_num_compatible(guess, true_sol):
    """
    Given guess and true_sol (arrays of integers of lenght WLEN) representing our first
    guess and a true solution, calculate how many of the candidate solutions are
    compatible with the gray-green-yellow pattern revealed by our first guess.
    """

    # Start with all solutions being allowed
    filt = np.ones(NS, dtype = 'bool')

    # Go through the letters
    for i in range(WLEN):

        # i-th letter is green -> only pick solutions with the same i-th letter
        if guess[i] == true_sol[i]:
            filt *= (solns[:, i] == true_sol[i])
        else:
            # i-th letter is yellow -> this letter must be present in the solution
            if np.count_nonzero(guess[i] == true_sol):
                filt *= letters_present[:, guess[i]]
            # i-th letter is gray -> this letter must be absent in the solution
            else:
                filt *= np.logical_not(letters_present[:, guess[i]])

    # Count the number of compatible solutions
    return np.sum(filt)


# For a given initial guess, for each possible solution calculate how many solutions are
# still allowed after revealing the first green-yellow-gray pattern
remaining = np.zeros(NS, dtype = int)

# For each initial guess, the expected number of solutions remaining after the first guess
remaining_ev = np.zeros(NW)

# To monitor the improvement
best = 1000

# Cycle through the initial guesses
for i in range(NW):

    # Keep track of progress
    if i % 500 == 0:
        print('epoch', i)

    # Cycle through solutions and calculate the expected number of solutions remaining
    # if we start the guessing with the NW-th word
    for j in range(NS):
        remaining[j] = get_num_compatible(wlist[i], solns[j])
    remaining_ev[i] = np.mean(remaining)

    # Keep track of the improvement
    if remaining_ev[i] < best:
        best = remaining_ev[i]
        print(f'{best:.3f}', ''.join([chr(ORD_A + c) for c in wlist[i]]))

# Store the results
np.savetxt('output.txt', remaining_ev)
