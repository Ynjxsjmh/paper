#include "chs.h"

CHSTree::CHSTree(vector<set<string> > set_cluster) {
    root->edge = "";
    root->set_cluster_before_simplified = set_cluster;
    root->set_cluster_after_simplified = simplifySetCuster(set_cluster);
    root->extension_set = getExtensionSet(set_cluster);

    generateTree(root);
}

CHSTree::~CHSTree() {
}


/**
 * 生成 CHS 树
 *
 * 1. 化简集合簇：去掉集合簇中的真超集，记化简后的集合簇为F。
 * 2. 找扩展节点：以F中势最小的集合作为扩展节点。如果存在多个集合的最小势相等，则找出集合中所有元素在集合簇F中出现频率之和最大的集合作为扩展节点。
 * 3. 对于势最小的集合CS＝{a1，a2，…，am｝，首先从集合簇F中删除包含a1的集合，记删除的元素为路径标签，返回步骤1，直到F变为空集，……，依次类比，直至删除F中包含am的集合。
 *
 * @param Node* 根节点
 */
void CHSTree::generateTree(Node* root) {
    if (root->set_cluster_before_simplified.size() <= 0) {
        return;
    }

    vector<set<string> > set_cluster = root->set_cluster_after_simplified;
    set<string> elements;

    for (string a : root->extension_set) {
        vector<set<string> > set_cluster_copied = set_cluster;

        elements.insert(a);
        set_cluster_copied = eraseElementsFromSetCuster(elements, set_cluster_copied);

        Node* child;
        child->edge = a;
        child->set_cluster_before_simplified = set_cluster_copied;
        child->set_cluster_after_simplified = simplifySetCuster(set_cluster_copied);
        child->extension_set = getExtensionSet(set_cluster_copied);

        root->children.push_back(child);

        generateTree(child);
    }
}

/**
 * 从集合簇中去除包含某些元素的集合
 *
 * @param elements 待去除的元素集合
 * @param set_cluster 集合簇
 * @return 去除后的集合簇
 */
vector<set<string> > CHSTree::eraseElementsFromSetCuster(set<string> elements, vector<set<string> > set_cluster) {
    for (string element : elements) {
        for(iterator it = begin; it != end(set_cluster);) {
            if (*it.count(element) > 0) {
                it = set_cluster.erase(it);
            } else {
                ++it;
            }
        }
    }

    return set_cluster;
}

/**
 * 化简集合簇
 *
 * 去掉集合簇中的真超集。
 *
 * 真超集定义：若集合S1和S2满足S2⊂S1,则称S1为S2的真超集。
 *             例如S1={a,b,c,d},S2={a,c},则S1是S2的真超集。
 *
 * @param set_cluster 集合簇
 * @return 化简后的集合簇
 */
vector<set<string> > CHSTree::simplifySetCuster(vector<set<string> > set_cluster) {
    vector<set<string> > set_cluster_after_simplified;

    for (int i = 0; i < set_cluster.size() - 1; i++) {
        // i 是否为 j 的超集
        int i_includes_j = 0;

        for (int j = 0; j < set_cluster.size(); j++) {
            if (includes(set_cluster[i].begin(), set_cluster[i].end(),
                         set_cluster[j].begin(), set_cluster[j].end()) &&
                i != j) {
                i_includes_j = 1;
                break;
            }
        }

        if (!i_includes_j) {
            set_cluster_after_simplified.push_back(set_cluster[i]);
        }
    }

    return set_cluster_after_simplified;
}


/**
 * 找出集合簇中势最小的集合
 *
 * 势的定义：该集合中元素的个数。
 *           例如集合A={a,b,c,d},则集合A的势为4。
 *
 * @param set_cluster 集合簇
 * @return 势最小的集合，由于集合可能有多个，返回一个 vector。
 */
vector<set<string> > CHSTree::getMinPotential(vector<set<string> > set_cluster) {
    int current_min = INT_MAX;
    vector<set<string> > min_potentials;

    for (set<string> s : set_cluster) {
        if (s.size() < current_min) {
            min_potentials.clear();
            min_potentials.push_back(s);
            current_min = s.size();
        } else if (s.size() > current_min) {
            ;
        } else {
            min_potentials.push_back(s);
        }
    }

    return min_potentials;
}


/**
 * 找出集合簇中的扩展集合
 *
 * 扩展集合：
 * 1. 如果势最小的集合只有一个，则为该集合
 * 2. 如果势最小的集合有多个，为集合元素在集合簇中出现频率之和最大的集合。
 *
 * 定义元素e在集合簇F中出现的频率：
 *    对于集合S，若元素e∈S，则称e与S相关联：
 *    集合簇F中与元素e相关联的集合的个数即为元素e在集合簇F中出现的频率。
 *
 * @param set_cluster 集合簇
 * @return 扩展集合
 */
set<string> CHSTree::getExtensionSet(vector<set<string> > set_cluster) {
    vector<set<string> > min_potentials = getMinPotential(set_cluster);

    if (min_potentials.size() == 1) {
        return min_potentials[1];
    } else {
        vector<int> frequencies;

        for (set<string> min_potential : min_potentials) {
            int frequency = 0;

            for (string e : min_potential) {
                for (int i = 0; i < set_cluster.size(); i++) {
                    if (set_cluster[i].count(e)) {
                        frequency++;
                    }
                }
            }

            frequencies.push_back(frequency);
        }

        int min_idx = min_element(frequencies.begin(), frequencies.end()) - frequencies.begin();

        return min_potentials[min_idx];
    }
}
