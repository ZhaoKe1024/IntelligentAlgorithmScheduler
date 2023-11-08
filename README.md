[Endlish](README.md)|[简体中文](README_cn.md)

# IntelligentAlgorithm-CloudScheduler

### Introduction
This project is dedicated to solving cloud computing scheduling problems using intelligent algorithms. It can replace evaluation functions, plot and store results and charts. It should be noted that all are discrete scheduling problems, which means that all particle swarm optimization algorithms are discrete particle swarm optimization algorithms.

In addition, the Optimization on Graph and Flexible Job-Shop Scheduling Problem are developing.

### Code Structureh

- simulate.py: The code can be run to obtain results, and algorithms, data, and parameters can be changed within it.
- SchedulerScaleandFitness.py: This file is used to compare the optimal solutions of multiple algorithms and their convergence curves.
- Schedulers.py: Similar to the above, it includes comparative experiments for different groups.
- chaosTest.py: Similar to the above, it includes comparative experiments for different groups.

utils.Entities : This file includes some entities that could tasks need.
- Cloudlet: cloud tasks to allocated
- VM: containers Virtual Machines to execute tasks(cloudlets)

schedulers/*.py, Support Algorithms to simulating.
Not every algorithm has a paper, as some are my own improvement attempts.
- GA: Genetic Algorithm
- SA: Simulated Annealing Algorithm
- ACO: Ant Colony Optimization Algorithm
- PSO: Particle Swarm Optimization Algorithm
- CRPSO: Chaotic Hierarchical Gene Replication
- DPSO: discrete PSO
- CDPSO: Chaotic PSO
- TSA: Taboo Search Algorithm
- others


- ChaosDPSO(DPSO based on Chaos)
- ChaosHPSO(DPSO based on Chaos and hierarchical)
- newPSO(modified PSO)

utils/*.py: tools 
The remaining codes that have not been introduced are utility classes or abandoned code from that year.


### Optimization on Graph
It mainly optimizes tasks on a Directed Acyclic Graph (DAG), and the most commonly used graph structure is task scheduling.
- ./GraphAlgorithm.py
- ./datastructure/ActivityGraph.py : Implementation of AOE and AOV(Activity on Vertex).
- ./datasets/graph_example/*.txt

### Topological Sorting
ActivityGraph.py : Implementing the AOV(Activity on Vertex) and AOE(Activity on Edge) Graph Model, and provides follow functions:
- AOV.topological_sort_all(self): Topological sorting of the graph.
- AOV.check_path(self, path)：determining whether a given array is a topological order (subsequence not implemented yet).
- AOE.critical_path(self): find the critical path on this AOE Graph.


#### Reference

1. Holland J.(1992). [Genetic Algorithm](https://doi.org/10.1038/scientificamerican0792-66)
2. Kennedy J, Eberhart R.(1995). Particle Swarm Optimization 
3. Ke Zhao.(2021). [Research on Edge Cloud Load Balancing Strategy based on Chaotic Hierarchical Gene Replication](https://www.fujipress.jp/jaciii/jc/jacii002600050758/)
