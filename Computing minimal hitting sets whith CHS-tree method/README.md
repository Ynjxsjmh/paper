
在命令行使用 `g++ main.cpp chs.cpp` 编译。

# CHS-tree 算法步骤

1. 化简集合簇：去掉集合簇中的真超集，记化简后的集合簇为 F。
2. 找扩展节点：以 F 中势最小的集合作为扩展节点。如果存在多个集合的最小势相等，则找出集合中所有元素在集合簇 F 中出现频率之和最大的集合作为扩展节点。
3. 对于势最小的集合 CS＝{a1，a2，…，am}，首先从集合簇 F 中删除包含 a1 的集合，记删除的元素为路径标签，返回步骤 1，直到 F 变为空集，……，依次类比，直至删除 F 中包含 am 的集合。
4. 从根节点到叶节点进行遍历，每条路径构成一个碰集，对其进行化简（去掉真超集），得到所有极小碰集。

# CHS-tree 算法实例

使用 CHS-tree 方法求集合簇 F={{2,4,5}, {1,2,3}, {1,3,5}, {2,4,6}, {2,4}, {2,3,5}, {1,6}}的所有极小碰集。

```
1) 化简集合簇：将 F 中为其他集合超集的集合删除,即删除{2,4,5}{2,4,6},因为它们是集合{2,4}的超集.剩余集合组成一个新的集合簇 F'={{1,2,3}, {1,3,5}, {2,4}, {2,3,5}, {1,6}}.
2) F'中势最小的集合有两个，分别为{2,4}和{1,6}.分别统计 F’中与元素 2,4,1,6 相关联的集合的个数为 3,1,3,1.由于元素 2 和 4 属于同一集合,与它们相关联的集合个数之和为 4;元素 1 和 6 也属于同一集合,与它们相关联的集合个数之和为 4.个数之和相等,因此从集合{2,4}和{1,6}中任意选一个集合作为根节点进行扩展,假设选择集合{2,4}.
3) 对于集合{2,4}，首先删除 F'中包含 2 的集合,剩余集合{1,3,5}, {1,6}构成一个新的集合簇 F1,回到步骤一
  3.1) 化简集合簇（略）,当前集合簇 F1 为{{1,3,5}, {1,6}}
  3.2) 选择 F1 中势最小的集合{1,6}进行扩展;
  3.3) 对于集合{1,6},删除 F1 中包含 1 的集合,集合簇变为空集,1 为终止节点;
  3.3’) 对于集合{1,6},删除 F1 中包含 6 的集合,剩余集合为{1,3,5};去掉元素 1(1 在左边碰集中已经出现. ,集合变为{3,5}, 回到步骤一
    3.3’.1) 化简集合簇（略）,当前集合簇为{{3,5}}
    3.3’.2) 选择势最小的集合{3,5}进行扩展;
    3.3’.3) 对于集合{3,5}，删除元素 3,集合簇为空集,3 为终止节点;
    3.3’.3’) 对于集合{3,5}，删除元素 5,集合簇为空集,5 为终止节点.
3’) 对于集合{2,4}，删除 F'中包含 4 的集合,集合簇为{{1,2,3}, {1,3,5}, {2,3,5}, {1,6}}, 去掉集合中的元素 2(2 和 4 在同一层,包含 2 的碰集已经求出. ,得到新的集合簇 F2={{1,3}, {1,3,5}, {3,5}, {1,6}}.回到步骤一
  3’.1) 化简集合簇: {1,3,5}为{1,3}的超集, 对 F2 去超集后为 F'2={{1,3}, {3,5}, {1,6}}。
  3’.2) 分别统计 F’2 与元素 1,3,5,6 相关联的集合的个数为 2,2,1,1.因为元素 1 和 3 属于同一集合,与它们相关联的集合个数之和为 4,和最大,所以对集合{1,3}进行扩展.
  3’.3) 对于集合{1,3},删除 F’2 中包含元素 1 的集合,剩余集合为{3,5}, 回到步骤一
    3’.2.1) 化简集合簇（略）,当前集合簇为{{3,5}}
    3’.2.2) 选择势最小的集合{3,5}进行扩展;
    3’.2.3) 对于集合{3,5},删除包含元素 3 的集合,集合簇为空集,故 3 为终止节点;
    3’.2.3’) 对于集合{3,5},删除元素包含元素 5 的集合,集合簇为空集,故 5 为终止节点;
  3’.3’) 对集合{1,3},去掉 F'2 中包含 3 的集合,剩余集合为{1,6}, 回到步骤一
    3’.3’.1) 化简集合簇（略）,当前集合簇为{{1,6}}
    3’.3'.2) 选择势最小的集合{1,6}进行扩展;
    3’.3’.3) 去掉元素 1(1 和 3 在同一层,包含 1 的碰集已经求出),得到集合{6},  回到步骤一
      3’.3’.3.1) 化简集合簇（略）,当前集合簇为{{6}}
      3’.3’.3.2) 选择势最小的集合{6}进行扩展;
      3’.3’.3.3) 对于集合{6},删除包含元素 6 的集合,集合簇变为空集,故 6 为终止节点.
4) 分别从根节点到叶节点进行遍历,每条路径上的元素组成一个集合,这些集合构成了所有极小碰集 MHSs,即{2,1}, {2,6,3}, {2,6,5}, {4,1,3}, {4,1,5}, {4,3,6}, 如下图所示.
```

![](./images/20201223_205639_RS0237.png)

从上图可以看出,用 CHS-tree 算法得到的所有极小碰集为{2,1}, {2,6,3}, {2,6,5}, {4,1,3}, {4,1,5}, {4,3,6}。