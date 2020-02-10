#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <ctime>
#include "metis.h"
using namespace std;

const double p_rate = 0.25;
const double eexp = 1e-8;

namespace Data {
    vector<vector<double> > features;
    vector<double> labels;

    // cum stores cumulative sum series and cumulative squared sum series
    // The first value is sigma(t_i) and the second is sigma(t_i^2)
    vector<vector<pair<double, double> > > cum;
}

namespace Algorithm {

    // partvec is the result of the division
    vector<int> partvec;

    // clusters stores clustering results
    vector<vector<int> > subclusters, clusters;

}

void load_Data(const string &file, vector<vector<double> > &features, 
                vector<double> &labels) {
    ifstream in_file(file, ios::in);

    string temp_line;
    // Line-by-line read
    while (getline(in_file, temp_line)) {
        istringstream iss(temp_line);
        string temp_str;
        vector<double> values;
        // Split each line string with a comma
        while (getline(iss, temp_str, ',')) {
            if (features.size() == labels.size()) {
                labels.push_back(stof(temp_str));
            } else {
                values.push_back(stof(temp_str));
            }
        }
        features.push_back(values);
    }

    return;
}

void calculate_cum(const vector<vector<double> > &features,
                   vector<vector<pair<double, double> > > &cum) {
    vector<pair<double, double> > temp_line;
    temp_line.resize(features[0].size() + 1, make_pair(0., 0.));
    cum.resize(features.size(), temp_line);

    for (int i = 0; i < features.size(); ++ i) {
        for (int j = 1; j <= features[0].size(); ++ j) {
            cum[i][j].first = cum[i][j - 1].first + features[i][j - 1];
            cum[i][j].second = cum[i][j - 1].second 
                               + features[i][j - 1] * features[i][j - 1];
        }
    }

    return;
}

/*
void discrete(const vector<double> &labels, vector<double> &unique_labels,
              vector<int> &label_index, vector<int> &label_count) {
    unique_labels = labels;
    sort(unique_labels.begin(), unique_labels.end());
    auto e = unique(unique_labels.begin(), unique_labels.end());
    unique_labels.erase(e, unique_labels.end());
    for (int i = 0; i < labels.size(); ++ i) {
        label_index[i] = lower_bound(unique_labels.begin(), e, labels[i]) - unique_labels.begin();
        ++ label_count[label_index[i]];
    }
    label_count.erase(label_count.begin() + unique_labels.size(), label_count.end());
}
*/

int count_labels(const vector<double> &labels) {
    auto unique_labels = labels;
    sort(unique_labels.begin(), unique_labels.end());
    auto e = unique(unique_labels.begin(), unique_labels.end());
    return (e - unique_labels.begin());
}

void make_grid(int left, int right, int down, int up, int m,
               vector<int> &wd_list, vector<int> &wl_list) {
    for (int i = left; i < right; ++ i) {
        wd_list.push_back(i);
    }
    for (int i = down; i < up; ++ i) {
        int wl = round(0.025 * (i + 1) * m);
        if (wl >= 10) wl_list.push_back(wl);
    }
    wl_list.erase(unique(wl_list.begin(), wl_list.end()), wl_list.end());
    
    return;
}

void indicating(const vector<vector<double> > &featrue, const int &wd,
                const int &wl, const int &symbolic_size,
                const vector<vector<pair<double, double> > > &cum,
                vector<vector<bool> > &indicating_array) {
    vector<bool> temp;
    temp.resize(symbolic_size);
    indicating_array.resize(featrue.size(), temp);
    
    for (int i = 0; i < featrue.size(); ++ i) {
        for (int j = 0; j < featrue[0].size() - wl + 1; ++ j) {
            double sumx = cum[i][j + wl].first - cum[i][j].first;
            double sumx2 = cum[i][j + wl].second - cum[i][j].second;
            double meanx = sumx / wl;
            double sigmax = sqrt(sumx2 / wl - meanx * meanx);
            
            // int length = 1. * wl / wd;
            // int ly = 0, ry = length;
            int val = 0, ly = 0, ry = 0;
            for (int k = 0; k < wd; ++ k) {
                ly = round(1. * wl / wd * k);
                ry = round(1. * wl / wd * (k + 1));
                double sumy = cum[i][j + ry].first - cum[i][j + ly].first;
                double meany = sumy / (ry - ly);
                double paa = (meany - meanx) / sigmax;
                int bit_val = (paa < 0.) ? 
                              ( (paa < -0.67) ? 0 : 1 ) : ( (paa < 0.67) ? 2 : 3 );
                val += (1 << (k << 1)) * bit_val;
                // ly = ry;
                // ry += length;
            }
            indicating_array[i][val] = true;
        }
    }

    return;
}

void find_candidates(const vector<vector<bool> > &indicating_array,
                     vector<int> &candidates, const int &symbolic_size,
                     const int &select_low, const int &select_high) {
    //  Count the number of occurrences of patterns
    vector<int> pattern_count(symbolic_size, 0);
    for (int i = 0; i < indicating_array.size(); ++ i) {
        for (int j = 0; j < symbolic_size; ++ j) {
            if (indicating_array[i][j]) ++ pattern_count[j];
        }
    }

    // Filter patterns that meet the requirements for the number of occurrences
    for (int i = 0; i < symbolic_size; ++ i) {
        if (pattern_count[i] > select_low && pattern_count[i] < select_high)
            candidates.push_back(i);
    }

    return;
}

void build_SPT(const vector<vector<bool> > &indicating_array, vector<int> &candidates,
               vector<vector<int> > &subclusters, const int &n, const int &label_size) {
    // Randomly rearrange candidates
    // select the pre-label_size as the classification feature
    shuffle(candidates.begin(), candidates.end(), default_random_engine(time(NULL)));

    vector<int> predict(n, -1);

    for (int i = 0; i < label_size - 1; ++ i) {
        int true_count = 0;
        for (int j = 0; j < n; ++ j) {
            if (indicating_array[j][candidates[i]]) ++ true_count;
        }
        if ((true_count << 1) <= n) {
            // When the number of trueist is less than false, 
            // the time series that contains the pattern is marked as the current classification. 
            // The rest is reserved for the next classification.
            for (int j = 0; j < n; ++ j) {
                if (indicating_array[j][candidates[i]] && predict[j] == -1)
                    predict[j] = i;
            }
        } else {
            // Conversely, the time series that does not contain the pattern 
            // is marked as the current classification.
            for (int j = 0; j < n; ++ j) {
                if (!indicating_array[j][candidates[i]] && predict[j] == -1)
                    predict[j] = i;
            }
        }
    }
    // The remaining time series is all categorized to the last category.
    for (int j = 0; j < n; ++ j) {
        if (predict[j] == -1) predict[j] = label_size - 1;
    }

    subclusters.push_back(predict);
    
    return;
}

void ensemble2graph(vector<vector<int> > &subclusters, const int &label_size,
                    vector<idx_t> &xadj, vector<idx_t> &adjncy) {
    xadj.push_back(0);
    int n = subclusters[0].size();

    // The result of each classification is independent.
    // Map the results of each classification to their corresponding intervals.
    for (int i = 0; i < subclusters.size(); ++ i) {
        for (int j = 0; j < n; ++ j) {
            subclusters[i][j] += n + label_size * i;
        }
    }

    // The first n point sits for n time series.
    // Add edges between points and categories.
    for (int j = 0; j < n; ++ j) {
        for (int i = 0; i < subclusters.size(); ++ i) {
            adjncy.push_back(subclusters[i][j]);
        }
        xadj.push_back(adjncy.size());
    }

    // The latter point corresponds to the categories.
    // Add edges to points in the same category.
    for (int i = 0; i < subclusters.size(); ++ i) {
        for (int c = 0; c < label_size; ++ c) {
            int v = n + label_size * i + c;
            for (int j = 0; j < n; ++ j) {
                if (subclusters[i][j] == v) {
                    adjncy.push_back(j);
                }
            }
            xadj.push_back(adjncy.size());
        }
    }

    return;
}

vector<vector<int> > get_subclusters(const vector<vector<double> > &features, 
                                     const int &n, const int &ensemble_size, 
                                     const int &wd, const int &wl,
                                     const int &symbolic_size, const int &label_size,
                                     vector<vector<pair<double, double> > > &cum) {
    // indicating_array[i][j] indicates whether the ith time series contains the jth symbol pattern
    vector<vector<bool> > indicating_array;
    indicating(features, wd, wl, symbolic_size, cum, indicating_array);

    // The upper and lower bounds of the count of symbol patterns that can be selected as features
    const double select_low = n * p_rate / label_size;
    const double select_high = n * (1 - p_rate / label_size);

    // candidates is a collection of selected symbol patterns
    vector<int> candidates;
    find_candidates(indicating_array, candidates, 
                    symbolic_size, select_low, select_high);
    
    vector<vector<int> > res;
    if (candidates.size() < label_size) return res;

    // Build ensemble_size tree
    for (int k = 0; k < ensemble_size; ++ k) {
        build_SPT(indicating_array, candidates, res, n, label_size);
    }

    return res;
}

vector<int> get_parts(vector<vector<int> > &clusters, const int &label_size,
                      const int &n) {
    // nvtxs is the number of vertices in the graph (equals to xadj.size() + 1)
    // ncon is the number of balancing constraints 
    idx_t nvtxs = n + clusters.size() * label_size,
          ncon = 1, nparts = label_size, objval = 0;
    // xadj and adjncy store the graph using the compressed storage format (CSR)
    // partvec stores the partition vector of the graph.
    vector<idx_t> xadj, adjncy, part(nvtxs, 0);

    // Turning forests into graphs
    ensemble2graph(clusters, label_size, xadj, adjncy);

    vector<int> res(n, 0);

    // Divide the graph using the metis component
    int flag = METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adjncy.data(), 
                                   0, 0, 0, &nparts, 0, 0, 0, &objval, part.data());
    if (flag != METIS_OK) {
        cout << "Fail to divide the graph." << endl;
        return res;
    }

    copy(part.begin(), part.begin() + n, res.begin());

    return res;
}

double rand_index(const vector<int> &predict, const vector<double> &real) {
    int n = predict.size();
    int tp = 0, fp = 0, tn = 0, fn = 0;
    for (int i = 0; i < n - 1; ++ i) {
        for (int j = i + 1; j < n; ++ j) {
            if (abs(real[i] - real[j]) < eexp) {
                if (predict[i] == predict[j]) ++ tp;
                else ++ fn;
            } else {
                if (predict[i] == predict[j]) ++ fp;
                else ++ tn;
            }
        }
    }
    return 1. * (tp + tn) / (tp + tn + fp + fn);
}


int main(int argc, char *argv[]) {
    string Dataset = string(argv[1]);
    const int ensemble_size = atoi(argv[2]);
    //    string Dataset; int ensemble_size;
    //    cin >> Dataset >> ensemble_size;
    // string Dataset = "ElectricDevices";
    // int ensemble_size = 100;
    cout << "Dataset: " << Dataset << ", ensemble size: " << ensemble_size << endl;
    
    string train_file = "./" + Dataset + "/" + Dataset + "_TRAIN";
    string test_file = "./" + Dataset + "/" + Dataset + "_TEST";
    
    // Because it's clustering, we don't need Data for training
    // Train varaibles contain both training and testing data from the UCR datasets
    load_Data(train_file, Data::features, Data::labels);
    load_Data(test_file, Data::features, Data::labels);

    if (Data::features.size() == 0) {
        cout << "Fail to read Data." << endl;
        return 0;
    }
    /*
    // Discrete the labels
    // unique_labels is a collection of the values of the label
    // label_index is the position where the original label corresponds to the unique_labels
    // label_size is the number of labels in the unique_labels
    // label_count is the count of labels in the unique_labels
    vector<int> label_index(n), label_count(n);
    vector<double> unique_labels;
    discrete(labels, unique_labels, label_index, label_count);
    */

    // label_size is the number of labels in the unique_labels
    int label_size = count_labels(Data::labels);
    
    clock_t start_time = clock();

    // n is the number of time series
    // m is the length of the time series
    int n = Data::features.size();
    int m = Data::features[0].size();

    // Calculate the cumulative sum series and cumulative squared sum series
    calculate_cum(Data::features, Data::cum);

    // wd is the collection of w (the number of segments)
    // wl is the collection of l (subsequence length)
    vector<int> wd_list, wl_list;
    make_grid(3, 8, 0, 40, m, wd_list, wl_list);

    for (int i = 0; i < wd_list.size(); ++ i) {
        int wd = wd_list[i];
        int symbolic_size = 1 << (wd << 1);

        for (int j = 0; j < wl_list.size(); ++ j) {
            int wl = wl_list[j];

            Algorithm::subclusters = get_subclusters(Data::features, n, ensemble_size,
                                    wd, wl, symbolic_size, label_size, Data::cum);
            if (Algorithm::subclusters.size() == 0) continue;

            Algorithm::partvec = get_parts(Algorithm::subclusters, label_size, n);
            Algorithm::clusters.push_back(Algorithm::partvec);
        }
    }

    // Collect clusters and do final ensemble
    Algorithm::partvec = get_parts(Algorithm::clusters, label_size, n);

    clock_t end_time = clock();
    
    // Calculating the Rand Index
    double rand_idx = rand_index(Algorithm::partvec, Data::labels);
    cout << "rand index: " << rand_idx << endl;
    cout << "The running time is: " << fixed << (double)(end_time - start_time) / CLOCKS_PER_SEC << "seconds" << endl;
    
    return 0;
}