import os
import itertools
import numpy as np


def ConnectInformation(filename):
    """
    这个函数用于抽取模块之间的连接关系

    Input:
    filename = "./Ports_link_input_<id>.txt"

    Output:
    edge_1.dat
    edge_2.dat
    n_edges.dat
    """
    n_edge = []   # 存取模块直接的连接关系
    list_m = []   # 临时变量
    list_m_port=[] #每组模块的连接关系中端口对应关系
    edge_1 = []   # 存取模块标号

    
    with open(filename) as f:
        #line = f.readline()
        while 1:
            line = f.readline()
            if not line:
                break
            if line.split()[0][0] == "M":
                macro_group=[]
                """n_edges.dat"""
                [list_m.append(int(value.lstrip("M"))-1) for value in line.split()]
                #n_edge.append(list_m)
                #print(list_m, '\n')
                """edge_1.dat"""
                tmp = list(itertools.combinations(list_m, 2))
                [edge_1.append(tmp[i][j]) for i in range(len(tmp)) for j in range(2)]
                #list_m = []
                #next line contains information about ports connection 
                line = f.readline()
                for value in line.split():
                    list_m_port.append(value)
                #print(list_m_port,'\n')
                for j in range(len(list_m)):    
                    macro_group.append([list_m[j], int(list_m_port[j])])
                n_edge.append(macro_group)
                
            list_m = [] 
            list_m_port=[]
            #line = f.readline()
            #if not line:
                #break

    """edge_2.dat"""
    edge_2 = np.zeros((len(edge_1)),dtype = int)   # 存取模块标号
    edge_2[::2] = edge_1[1::2]
    edge_2[1::2] = edge_1[0::2]
    edge_2 = list(edge_2)

    """Write into dat file"""
    # if not os.path.exists("./SJTU_connect"):
        # os.mkdir("./SJTU_connect")

    out_filename0 = "n_edges.dat"
    out_filename1 = "edges_1.dat"
    out_filename2 = "edges_2.dat"
    string_f0 = os.path.join("./",out_filename0)
    string_f1 = os.path.join("./",out_filename1)
    string_f2 = os.path.join("./",out_filename2)
    f.close()

    with open(string_f0, "w") as f0:
        f0.write(str(n_edge))

    with open(string_f1, "w") as f1:
        f1.write(str(edge_1))

    with open(string_f2, "w") as f2:
        f2.write(str(edge_2))
    
    return True

def TrainingData(area_filename, link_filename):
    #macro_num = int(area_filename.split("/")[-2].split("-")[0])     
    #os.system(f"cp {area_filename} ./InputDataSample/DataSet/Ports_area_etc_input_{case_id}.txt")
    #os.system(f"cp {link_filename} ./InputDataSample/DataSet/Ports_link_input_{case_id}.txt")
    os.system(f"cp {area_filename} ./placement_info.txt")     #for Valid_list_gen.py
    #link_file = f"./InputDataSample/DataSet/Ports_link_input_{case_id}.txt"
    #macro_num = int(area_filename.split("/")[-2].split("-")[0])
    """  get case_id   """
    case_id=(area_filename.split('_')[-1]).split('.')[0]
    print(case_id)
    """ get macro numbers"""
    f=open(area_filename, 'r')
    macro_num=0
    file_list=f.readlines()
    for i in range(len(file_list)):
        if("Module"==file_list[i].split(":")[0]):
            macro_num=macro_num+1
    print(macro_num)
    """ make three data files for GCN and HPWL """
    if(True==ConnectInformation(link_filename)):
        if(os.path.exists('data')):      #folder ./data exists, remove it
            os.system("rm -rf data")
        os.system("mkdir data")
        os.system("mv edges_1.dat data/edges_1.dat")
        os.system("mv edges_2.dat data/edges_2.dat")
        os.system("mv n_edges.dat data/n_edges.dat")
        return macro_num, case_id 