#include <stdio.h>

void print_rhombus_pattern(int n) {
    int i, j;
    for (i = 0; i < n; i++) 
    {
        for (j = 0; j < n - i - 1; j++)
         {
            printf(" ");
        }
        for (j = 0; j < i + 1; j++)
         {
            printf("/");
        }

        for (j = 0; j < i + 1; j++)
         {
            printf("\\");
        }
        printf("\n");
    }
    for (i = 0; i < n; i++) {
        for (j = 0; j < i + 1; j++) {
            printf(" ");
        }

        for (j = 0; j < n - i - 1; j++) {
            printf("\\");
        }

        for (j = 0; j < n - i - 1; j++) {
            printf("/");
        }

        printf("\n");
    }
}

int main() {
    int n;
    scanf("%d", &n);
    print_rhombus_pattern(n);

    return 0;
}
