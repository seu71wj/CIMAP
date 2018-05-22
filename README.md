# CIMAP
KDD'18 (Class-IMbalance Aware Partial Label Learning)
***************************************************************************
MATLAB toolbox for the Class-IMbalance Aware Partial label learning algorithm CIMAP.

[Contributor]
Jing Wang (jing_w@seu.edu.cn),
Min-Ling Zhang (zhangml@seu.edu.cn)
School of Computer Science and Engineering, Southeast University, Nanjing 210096, China

This code package can be used freely for academic, non-profit purposes. For other usage, please contact us for further information (Prof. Min-Ling Zhang: zhangml@seu.edu.cn ).
***************************************************************************

1.Introduction
===========================================================================
This package implements the class-imbalance aware approach for partial label (PL) learning named CIMAP [1]. This code package consists of two joint components: the data-level processing component which transforms the original PL training set by disambiguation and replenishment, and the model training component which learns the predictive model from the transformed data set with specified PL learning algorithm.

In this code pacakge, IPAL [2] is specified as the PL learning algorithm to serve as the model training component. It is worth noting that users can freely replace IPAL with other PL learning algorithm to instantiate CIMAP.

[1] J. Wang, M.-L. Zhang. Towards mitigating the class-imbalance problem for partial label learning. In: Proceedings of the 24th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD'18), London, UK, 2018, in press.

[2] M.-L. Zhang, F. Yu. Solving the partial label learning problem: An instance-based approach. In: Proceedings of the 24th International Joint Conference on Artificial Intelligence (IJCAI'15), Buenos Aires, Argentina, 2015, 4048-4054.

2. Requirements
===========================================================================
- Matlab, version 2014a and higher.

3. Installation
===========================================================================
- Unzip the code package and add the unzipped directory to your Matlab path

4. How to start?
===========================================================================
To illustrate the usage of CIMAP, we have included a demo (c.f. demo.m) in this code package. Note again that IPAL can be replaced by any other PL learnig algorithm for inducing the predictive model.
