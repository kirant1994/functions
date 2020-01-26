# Author : Kiran Praveen
import numpy as np

# Collapses repeating characters in a sequence.
# Can be used to simulate sample outputs for CTC loss.
# Blank character is removed from the final output
def collapse_sequence(arr, blank=None):
	if len(arr) == 0:
		return arr
	arr_collapsed = [arr[0]]
	prev_item = arr_collapsed[0]
	for item in arr[1:]:
		if item != prev_item and item != blank:
			arr_collapsed.append(item)
		prev_item = item
	return arr_collapsed

if __name__ == '__main__':
	arr = [1, 1, 1, 0, 2, 2, 0, 4, 0, 2, 2, 2, 3, 3, 3, 1, 1]
	print(arr)
	print(collapse_sequence(arr, blank=0))
