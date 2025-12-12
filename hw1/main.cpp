#include <iostream>
#include <cmath>
#include <iomanip>
using namespace std;

int main() {
    double a, b, c;
    cin >> a >> b >> c;

    if (a == 0 && b == 0 && c == 0) {
        cout << "any";
        return 0;
    }
    if (a == 0 && b == 0) {
        cout << "incorrect";
        return 0;
    }
    if (a == 0) {
        double x = -c / b;
        cout << fixed << setprecision(6) << x;
        return 0;
    }

    double D = b * b - 4 * a * c;

    if (D > 0) {
        double sqrtD = sqrt(D);
        double x1 = (-b + sqrtD) / (2 * a);
        double x2 = (-b - sqrtD) / (2 * a);
        cout << fixed << setprecision(6) << x1 << " " << x2;
    } else if (D == 0) {
        double x =  -b / (2 * a);
        cout << fixed << setprecision(6) << x;
    } else {
        cout << "imaginary";
    }

    return 0;
}
