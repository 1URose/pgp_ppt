#include <iostream>
#include <iomanip>
using namespace std;

int main() {
    int n;
    cin >> n;

    float *arr = new float[n];

    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }

    for (int i = 1; i < n; i++) {
        float key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }

    for (int i = 0; i < n; i++) {
        cout << scientific << arr[i];
        if (i + 1 < n) cout << " ";
    }
    cout << "\n";

    delete[] arr;
    return 0;
}
