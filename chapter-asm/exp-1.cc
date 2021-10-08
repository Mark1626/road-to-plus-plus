#include <stdio.h>

int mul1(int a, int b) {
    int val;
    __asm__ __volatile__("imull  %%ebx,%%eax"
                             :"=a"(val)
                             :"a"(a), "b"(b)
                             );

    return val;
}

int mul2(int a, int b) {
    int val;
    __asm__ __volatile__(   "movl  %1, %0\n\t"
                            "imull %2, %0"
                        :   "=r"(val)
                        :   "r"(a), "r"(b)
                        );

    return val;
}

int feeling_lucky(int a, int b, int c) {
    __asm__ __volatile__(   "mov    %1, %0\n\t"
                            "imull  %2, %0\n\t"
                            "imull  %3, %0\n\t"
                        :   "=r"(c)
                        :   "r"(a), "r"(b), "r"(c)
                        );
    return c;
}

int main() {
    int a = 7865;
    int b = 1234;

    printf("%d * %d = %d\n", a, b, mul1(a, b));

    printf("%d * %d = %d\n", a, b, mul2(a, b));

    int c = 100;
    printf("%d * %d * %d = %d %d\n", a, b, c, feeling_lucky(a, b, c), c);
}
