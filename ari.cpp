#include <iostream>
#include <vector>
using namespace std;

void discrete(const vector<double> &labels, 
              vector<int> &label_index, vector<int> &label_count) {
    auto unique_labels = labels;
    sort(unique_labels.begin(), unique_labels.end());
    auto e = unique(unique_labels.begin(), unique_labels.end());
    unique_labels.erase(e, unique_labels.end());
    label_index.resize(labels.size(), 0);
    label_count.resize(unique_labels.size(), 0);
    for (int i = 0; i < labels.size(); ++ i) {
        label_index[i] = lower_bound(unique_labels.begin(), e, labels[i]) - unique_labels.begin();
        ++ label_count[label_index[i]];
    }
    label_count.erase(label_count.begin() + unique_labels.size(), label_count.end());
}

int combination2(int n) {
    return n * (n - 1) / 2;
}

double get_ari(const vector<double> &pred, const vector<double> &real) {
    vector<int> li_pred, li_real, lc_pred, lc_real;
    discrete(pred, li_pred, lc_pred);
    discrete(real, li_real, lc_real);

    int m = pred.size();
    int n = lc_pred.size();
    vector<vector<int> > cnt(n, vector<int>(n));

    for (int i = 0; i < m; ++ i) {
        ++ cnt[li_pred[i]][li_real[i]];
    }
    
    int sum_comb_ij = 0;
    for (int i = 0; i < n; ++ i) {
        for (int j = 0; j < n; ++ j) {
            if (cnt[i][j] > 1) 
                sum_comb_ij += combination2(cnt[i][j]);
        }
    }

    int sum_comb_i = 0, sum_comb_j = 0;
    for (int i = 0; i < n; ++ i) {
        if (lc_pred[i] > 1) sum_comb_i += combination2(lc_pred[i]);
        if (lc_real[i] > 1) sum_comb_j += combination2(lc_real[i]);
    }

    double e = 1. * sum_comb_i * sum_comb_j / combination2(m);
    double ari = (sum_comb_ij - e) / (0.5 * (sum_comb_i + sum_comb_j) - e);
    
    return ari;
}

int main() {
    vector<double> pred({1, 1, 2, 2, 2, 3, 2, 4});
    vector<double> real({1, 1, 2, 2, 2, 3, 2, 2});



    cout << get_ari(pred, real) << endl;

    return 0;
}