// C++ program to implement 
// iterative Binary Search 
#include <bits/stdc++.h> 
using namespace std; 

// Iretative Binary Search function to find the index of an 
// element 'x' in a sorted array 'arr' if elements is 
// present, otherwise it return -1 

// low: The index of the first element in the current 
// sub-array high: The index of the last element in the 
// current sub-array 
int binarySearch(int arr[], int low, int high, int x) 
{ 
	while (low <= high) { 
		int mid = low + (high - low) / 2; 

		// If the middle element is equal to 'x', we have 
		// found the element, return its index 
		if (arr[mid] == x) 
			return mid; 

		// If the middle element is smaller than 'x', search 
		// in the right half of the array 
		if (arr[mid] < x) 
			low = mid + 1; 

		// If the middle element is greater than 'x', search 
		// in the left half of the array 
		else
			high = mid - 1; 
	} 

	// If the base case is reached, the element is not 
	// present in the array, return -1 
	return -1; 
} 

// Driver code 
int main(void) 
{ 
	int arr[] = { 2, 3, 4, 10, 40 }; 

	// Element to be searched 
	int x = 10; 
	int n = sizeof(arr) / sizeof(arr[0]); 
	int result = binarySearch(arr, 0, n - 1, x); 
	(result == -1) 
		? cout << "Element is not present in array"
		: cout << "Element is present at index " << result; 
	return 0; 
}
