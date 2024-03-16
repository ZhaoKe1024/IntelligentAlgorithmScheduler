#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/12/5 9:30
# @Author: ZhaoKe
# @File : expe_densegantt.py
# @Software: PyCharm
import pandas as pd
from fjspkits.fjsp_entities import Machine, Task
from fjspkits.FJSP_GAModel import check_toposort
from eautils.plottools import plot_gantt

def read_result(res_path="./results/"):
    machines = []
    job_set = set()
    # processing_time = 0
    idle_time = 0
    makespan = 0
    fname = "sdej20m10remove1/sde_20j10m_0000"
    verion = ""
    with open(res_path+fname+verion+".txt") as fin:
        line_num = 0
        line_str = fin.readline()
        tsk_cnt = 0
        while line_str:
            # line_parts = line_str.split('||')[:-1]
            line_parts = line_str.split('|')[:-1]
            tmp_m = Machine(0)
            last_time = 0
            for tsk in line_parts:
                jid_part = tsk.split('-')
                job_id = int(jid_part[0][1:])  # 19
                ope_part = jid_part[1].split(',')  # 3)[209.0 , 217.0]
                job_set.add(job_id)
                ope_id = ope_part[0].split(')')  # 3 , [209.0
                tsk_tmp = Task(tsk_cnt, int(job_id), int(ope_id[0]))
                # end = tsk.index(',')
                st = float(ope_id[1][1:])
                et = float(ope_part[1][:-1])
                # print(f"Task({int(tsk[5])}-{int(tsk[7])})[{int(tsk[10:end])}, {int(tsk[end+1:-1])}]||", end=', ')
                tsk_tmp.set_duration(st, et)
                tsk_tmp.selected_time = et-st
                tmp_m.add_task(tsk_tmp)
                idle_time += st - last_time
                last_time = et
                tsk_cnt += 1
            makespan = max(makespan, tmp_m.task_list[-1].finish_time)
            line_num += 1
            machines.append(tmp_m)
            line_str = fin.readline()
            # print()
    isvalid = check_toposort(machines, job_num=len(job_set), machine_num=len(machines))
    if isvalid:
        print("---legal schedule! ")
        print("makespan:", makespan)
    else:
        print("--------illegal schedule-----!")
        return
    # 根据machines得到一个pandas用于绘图
    data_dict = {"Task": {}, "Machine": {}, "Job": {}, "start_num": {}, "end_num": {}, "days_start_to_end": {}}
    for idx, machine in enumerate(machines):
        for task in machine.task_list:
            # 修改了这个地方的机器编号，因为我发现有时候甘特图和结果对不上，看来是Task的selected_machine有误，没有正确赋值，还需要检查
            data_dict["Machine"][task.global_index] = "M" + str(idx)
            data_dict["Task"][task.global_index] = f"Task[{task.parent_job}-{task.injob_index}]"
            data_dict["Job"][task.global_index] = "Job" + str(task.parent_job)
            data_dict["start_num"][task.global_index] = task.start_time
            data_dict["end_num"][task.global_index] = task.finish_time
            data_dict["days_start_to_end"][task.global_index] = task.selected_time
    df = pd.DataFrame(data_dict)
    plot_gantt(df, line_num, fname="./results/"+fname+verion+".png")
    with open("./results/"+fname+verion+"-res.txt", 'w') as fout:
        # fout.write(f"\nisvalid:{isvalid}, makespan:{makespan}, idletime:{idle_time}")
        fout.write(f"\nmakespan:{makespan}, idletime:{idle_time}")
    print("end")


if __name__ == '__main__':
    read_result()
