#include <iostream>
using namespace std;
int main()
{
	char op;
	float num1, num2;
	cin >> op;
	cin >> num1 >> num2;
	switch (op) {
	case '+':
		cout << num1 + num2;
		break;

	// If user enter -
	case '-':
		cout << num1 - num2;
		break;

	// If user enter *
	case '*':
		cout << num1 * num2;
		break;

	// If user enter /
	case '/':
		cout << num1 / num2;
		break;

	// If the operator is other than +, -, * or /,
	// error message will display
	default:
		cout << "Error! operator is not correct";
	}
	// switch statement ends

	return 0;
}
