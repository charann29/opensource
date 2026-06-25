#include <stdio.h>
#include <string.h>
char input[100];
int i = 0, error = 0;
void E();
void T();
void Eds();
void Tds();
void F();
int main() {
    printf("Recursive descent parsing for grammar\n");
    printf("\n$(id*id)+id\n\n");
    fgets(input, 100, stdin);

    E();

    if (input[i] == '\0' && error == 0) {
        printf("String is Accepted");
    } else {
        printf("String rejected");
    }

    return 0;
}

void E() {
    T();
    Eds();
}

void T() {
    F();
    Tds();
}

void Eds() {
    if (input[i] == '+') {
        i++;
        T();
        Eds();
    }
}

void Tds() {
    if (input[i] == '*') {
        i++;
        F();
        Tds();
    }
}

void F() {
    if (input[i] == '(') {
        i++;
        E();
        if (input[i] == ')') { // added to ensure closing parenthesis is handled
            i++;
        } else {
            error = 1;
        }
    } else if (input[i] == 'i') {
        i++;
    } else {
        error = 1;
    }
}
