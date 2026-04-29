#include <stdio.h>
#include "circle.h"

int main() {
    double r;
    printf("Please enter the radius of the circle: ");
    if (scanf("%lf", &r) != 1) {
        printf("Invalid input. Please enter a number.\n");
        return 1;
    }
    
    double area = calculate_area(r);
    printf("The area of a circle with radius %.2f is: %.2f\n", r, area);
    
    return 0;
}