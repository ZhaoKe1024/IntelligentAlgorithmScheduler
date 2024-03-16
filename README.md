[Endlish](README.md)|[简体中文](README_cn.md)

# Intelligent Algorithm Scheduler

# Introduction
This repository provides two models currently.

1. Solving FJSP using Intelligent Algorithm, and draw gantt plot.

``` python ./JSSP_executor.py ```

![](t202311201228_gantt.png)

![](t202311201228_iterplot.png)

2. This project is dedicated to solving cloud computing scheduling problems using intelligent algorithms. It can replace evaluation functions, plot and store results and charts. It should be noted that all are discrete scheduling problems, which means that all particle swarm optimization algorithms are discrete particle swarm optimization algorithms.
In addition, the Optimization on Graph and Flexible Job-Shop Scheduling Problem are developing.

``` python ./simulate.py ```

# Code Structure
```
root
└─JSSP_executor.py
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

### Example

``` python ./JSSP_executor.py ```

``` python ./simulate.py ```


#### Reference

1. Holland J.(1992). [Genetic Algorithm](https://doi.org/10.1038/scientificamerican0792-66)
2. Kennedy J, Eberhart R.(1995). Particle Swarm Optimization 
3. Ke Zhao.(2021). [Research on Edge Cloud Load Balancing Strategy based on Chaotic Hierarchical Gene Replication](https://www.fujipress.jp/jaciii/jc/jacii002600050758/)
4. [Job_Shop_Schedule_Problem](https://github.com/mcfadd/Job_Shop_Schedule_Problem)
