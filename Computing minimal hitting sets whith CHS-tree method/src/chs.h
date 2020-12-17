#include <string>
#include <set>
#include <vector>
#include <algorithm>


using namespace std;


class Node {
public:
    string edge;
    vector<set<string> > set_cluster_before_simplified;
    vector<set<string> > set_cluster_after_simplified;
    set<string> extension_set;
    vector<Node*> children;
};


class CHSTree {
public:
    CHSTree(vector<set<string> > set_cluster);
    ~CHSTree();

    void visualize(string filename);

private:
    void generateTree(Node* root);
    void destroyTree(Node* node);

    void write2dot(std::ofstream& outfile);
    void writeNode(std::ofstream& outfile, Node* curNode, int curNodeID);
    void writeEdge(std::ofstream& outfile, vector<Node*> curNodes, int curNodeID);

    void printSet(set<string> s, string ending);
    void printSetCluster(vector<set<string> > set_cluster);
    void printNode(Node* node);

    vector<set<string> > simplifySetCuster(vector<set<string> > set_cluster);
    vector<set<string> > getMinPotential(vector<set<string> > set_cluster);
    set<string> getExtensionSet(vector<set<string> > set_cluster);
    vector<set<string> > eraseSetContainsElementFromSetCuster(string element, vector<set<string> > set_cluster);
    vector<set<string> > eraseElementsFromSetCuster(set<string> elements, vector<set<string> > set_cluster);
    Node *root;
};
