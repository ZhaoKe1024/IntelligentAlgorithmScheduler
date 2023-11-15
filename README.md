[Endlish](README.md)|[简体中文](README_cn.md)

# Intelligent Algorithm Scheduler

# Introduction
This repository provides two models currently.

1. This project is dedicated to solving cloud computing scheduling problems using intelligent algorithms. It can replace evaluation functions, plot and store results and charts. It should be noted that all are discrete scheduling problems, which means that all particle swarm optimization algorithms are discrete particle swarm optimization algorithms.
In addition, the Optimization on Graph and Flexible Job-Shop Scheduling Problem are developing.

2. Solving FJSP using Intelligent Algorithm, and draw gantt plot.


# Code Structure
```
root
└─JSSP_executor.py
└─fjspkits/
│    └─fjsp_entities.py: Definition of Job, Task, Machine
│    └─fjsp_utils.py: Definition of some functions such as: read_file, calculate execute time.
│    └─FJSP_GAModel.py: flow of genetic algorithm based on POX Crossover for FJSP model.
└─simulate.py: For Cloud Scheduling, This code can be run to obtain results, and algorithms, data, and parameters can be changed within it.
└─schedulers/
│    └─GAScheduler.py, Genetic Algorithms to Cloud Service Scheduling.
│    └─DPSOTaskScheduler.py, Particle Swarm Algorithms to simulating.
│    └─SAScheduler.py, Simulating Anneal Algorithms to simulating.
│    └─ACScheduler.py, Ant Colony Algorithms to simulating.
│    └─TabooSearchScheduler.py, Taboo Search Algorithms to simulating.
│    └─*.py, Support Algorithms to simulating.
└─utils/
│    └─Entities : This file includes some entities that could tasks need. such as Cloudlet(cloud tasks to allocated), VM(containers Virtual Machines to execute tasks(cloudlets)).
│    └─plottools.py: some functions for plotting, such gantt plot.
└─GraphAlgorithm.py: AOE and AOV algorithm, such as critical path, topological sorting.
└─datastructure/
│    └─ActivityGraph.py: Definition of AOE(activities on Edges) and AOV(activities on Vertices) Model.
│    └─ *.py: Definitions of Entities or basic datastructures for Graph Model.
└─SchedulerScaleandFitness.py: This file is used to compare the optimal solutions of multiple algorithms and their convergence curves.
└─Schedulers.py: Similar to the above, it includes comparative experiments for different groups.
└─chaosTest.py: Similar to the above, it includes comparative experiments for different groups.
```

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
4. [Job_Shop_Schedule_Problem](https://github.com/mcfadd/Job_Shop_Schedule_Problem)
