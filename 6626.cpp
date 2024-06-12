#include <iostream>
#include <stack>
#include <unordered_map>
#include <string>

bool isValidParentheses(std::string s) {
    std::stack<char> parentheses_stack;
    std::unordered_map<char, char> matching_parentheses = {{')', '('}, {']', '['}, {'}', '{'}};

    for (char c : s) {
        if (matching_parentheses.find(c) != matching_parentheses.end()) {
            char top_element = parentheses_stack.empty() ? '#' : parentheses_stack.top();
            if (top_element != matching_parentheses[c]) {
                return false;
            }
            parentheses_stack.pop();
        } else {
            parentheses_stack.push(c);
        }
    }

    return parentheses_stack.empty();
}

int main() {
    std::string test_string = "{[()]}";
    if (isValidParentheses(test_string)) {
        std::cout << "Valid parentheses" << std::endl;
    } else {
        std::cout << "Invalid parentheses" << std::endl;
    }
    return 0;
}
