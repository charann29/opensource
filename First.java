// Java program to find factorial 
// of given number 

// Driver Class 
class Test { 
	// Method to find factorial 
	// of given number 
	static int factorial(int n) 
	{ 
		int res = 1, i; 
		for (i = 2; i <= n; i++) 
			res *= i; 
		return res; 
	} 

	// main method 
	public static void main(String[] args) 
	{ 
		int num = 5; 
		System.out.println("Factorial of " + num + " is "
						+ factorial(5)); 
	} 
}
