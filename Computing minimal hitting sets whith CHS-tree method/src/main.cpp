#include <iostream>
#include <string>
#include <set>
#include <vector>
#include <random>
#include <algorithm>

#include "chs.h"

using namespace std;


void printSetCluster(vector<set<string> > set_cluster) {
    cout << "{";

    for (size_t i = 0; i < set_cluster.size(); i++) {
        cout << (i ? ", " : "") << "{";

        set<std::string>::iterator setIt = set_cluster[i].begin();
        for (size_t j = 0; j < set_cluster[i].size(); j++) {
            cout << (j ? ", " : "") << *setIt;
            setIt++;
        }

        cout << "}";
    }

    cout << "}" << endl;
}

void cal(vector<set<string> > set_cluster) {
    CHSTree* chs = new CHSTree(set_cluster);

    cout << "Calculate the minimal hitting sets of following set cluster:" << endl;
    printSetCluster(set_cluster);

    chs->visualize("result");

    delete chs;
}

void demo() {
    vector<set<string> > demo {
        {"2", "4", "5"},
        {"1", "2", "3"},
        {"1", "3", "5"},
        {"2", "4", "6"},
        {"2", "3", "5"},
        {"2", "4"},
        {"1", "6"}
    };

    cal(demo);
}

void userInput() {
    vector<set<string> > set_cluster;

    cout << "Please input your set cluster:" << endl;
    cout << "1. Each line is treated as a set. Set elements should be seperated by comma." << endl;
    cout << "2. Line matches 'end' quits the input." << endl;
    string input = "";

    while (true) {
        cin >> input;

        set<string> s;

        string delimiter = ",";

        size_t pos = 0;
        string token;
        while ((pos = input.find(delimiter)) != string::npos) {
            token = input.substr(0, pos);
            s.insert(token);
            input.erase(0, pos + delimiter.length());
        }
        s.insert(input);

        if (input == "end") {
            break;
        } else {
            set_cluster.push_back(s);
        }
    }

    cal(set_cluster);
}

void randomOne() {
    vector<set<string> > set_cluster;
    int cluster_size;
    int min_val;
    int max_val;
    int max_num;

    cout << "How many sets do you want to generate: ";
    cin >> cluster_size;

    cout << "What's your data range (space seperated): ";
    cin >> min_val >> max_val;

    cout << "What's the maximum number of elements in a set: ";
    cin >> max_num;

    for (int i = 0; i < cluster_size; i++) {
        set<string> s;

        random_device                  rand_dev;
        mt19937                        generator(rand_dev());
        uniform_int_distribution<int>  value_distr(min_val, max_val);
        uniform_int_distribution<int>  set_distr(1, max_num);

        int set_size = set_distr(generator);

        while (s.size() < set_size) {
            s.insert(to_string(value_distr(generator)));
        }

        set_cluster.push_back(s);
    }

    cal(set_cluster);
}

void menu() {
    bool cont = true;

    while (cont) {
        cout << "===================" << endl;
        cout << "| 1. demo"           << endl;
        cout << "| 2. random"         << endl;
        cout << "| 3. user input"     << endl;
        cout << "| 4. quit"           << endl;
        cout << "===================" << endl;

        int choice;

        cout << "Input your choice: ";
        cin >> choice;

        switch (choice) {
        case (1):
            demo();
            break;
        case (2):
            randomOne();
            break;
        case (3):
            userInput();
            break;
        case (4):
            cont = false;
            break;
        }
    }
}

int main() {
    menu();
}
