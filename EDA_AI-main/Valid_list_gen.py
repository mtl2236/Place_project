"""Guangxi"""
def read_file(filename, r_flag, grid_size_n):
    """
    Input:
    filename = "./sample50_compact/50-1/placement_info.txt"
    r_flag = 1 补全矩形
    r_flag = 0 不补全矩形
    所有的面积 长度都按照矩形计算的

    Output:
    一个列表，列表里面的每一个元素是对应模块的一个字典，字典里面抽取了相关模块的信息

    """
    X_factor=0
    Y_factor=0
    origin_point=[]
    canvas_x_max=0
    canvas_x_min=0
    canvas_y_max=0
    canvas_y_min=0
    canvas_range=[]
    #whole_area=0
    m_list = []
    m_dict = {}
    with open(filename) as f:
        line = f.readline()
        while 1:
            flag = 0
            if line.split(":")[0] == "Area":
                canvas_b_list=[]
                canvas_x_list=[]
                canvas_y_list=[]
                for element in line.split("(")[1:]:
                    canvas_x_list.append(float(element.split(",")[0]))
                    canvas_y_list.append(float(element.split(",")[1].split(")")[0].strip()))
                    canvas_b_list.append([float(element.split(",")[0]), float(element.split(",")[1].split(")")[0].strip())])
                    canvas_x_max = max(canvas_x_list)
                    
                    canvas_x_min = min(canvas_x_list)
                    
                    canvas_y_max = max(canvas_y_list)
                    
                    canvas_y_min = min(canvas_y_list)
                    
                canvas_range.append(canvas_x_max)
                canvas_range.append(canvas_x_min)
                canvas_range.append(canvas_y_max)
                canvas_range.append(canvas_y_min)
                Canvas_Length_x = abs(canvas_x_max - canvas_x_min)
                Canvas_Length_y = abs(canvas_y_max - canvas_y_min)
                origin_point=canvas_b_list[0]
                """Guangxi"""
                X_factor=Canvas_Length_x/grid_size_n
                Y_factor=Canvas_Length_y/grid_size_n
            if line.split(":")[0] == "Module":
                id = line.split("M")[-1]
                m_dict[f"Module ID"] = int(id)-1
                p_id = 1
                while 1:
                    line = f.readline()

                    if line.split(":")[0] == "Boundary":
                        b_list = []
                        x_list = []
                        y_list = []
                        """Boundary Edge Extraction"""
                        for element in line.split("(")[1:]:
                            x_list.append(float(element.split(",")[0]))
                            y_list.append(float(element.split(",")[1].split(")")[0].strip()))
                            b_list.append([float(element.split(",")[0]), float(element.split(",")[1].split(")")[0].strip())])
                            x_max = max(x_list)
                            x_min = min(x_list)
                            y_max = max(y_list)
                            y_min = min(y_list)  

                        if r_flag == 1 and len(b_list) != 4:
                            b_list = []
                            b_list = [[x_min,y_min], [x_min,y_max], [x_max,y_max], [x_max,y_min]]     

                        """Length_x, Length_y Calculation, regard as rectangular"""
                        Length_x = abs(x_max - x_min)
                        Length_y = abs(y_max - y_min)
                        Central_x = (x_max + x_min) / 2
                        Central_y = (y_max + y_min) / 2
                        #M_area = Length_x * Length_y 

                        #m_dict["Boundary"] = b_list
                        #m_dict["Boundary_inf"] = line.split(";")[-1].strip("\n")
                        #m_dict["Module_Area"] = M_area
                        m_dict["Module_Lx"] = Length_x
                        m_dict["Module_Ly"] = Length_y
                        # m_dict["Module_Central_x"] = Central_x
                        # m_dict["Module_Central_y"] = Central_y
                        # m_dict["Left_down_x"]=-0.5*Length_x
                        # m_dict["Left_down_y"]=-0.5*Length_y
                        # m_dict["Left_up_x"]=-0.5*Length_x
                        # m_dict["Left_up_y"]=0.5*Length_y
                        # m_dict["Right_up_x"]=0.5*Length_x
                        # m_dict["Right_up_y"]=0.5*Length_y
                        # m_dict["Right_down_x"]=0.5*Length_x
                        # m_dict["Right_down_y"]=-0.5*Length_y

                    elif line.split(":")[0] == "Port":

                        p_list = []
                        px_list = []
                        py_list = []
                        """Port Edge Extraction"""
                        for element in line.split("(")[1:]:
                            p_list.append([float(element.split(",")[0]), float(element.split(",")[1].split(")")[0].strip())])
                            px_list.append(float(element.split(",")[0]))
                            py_list.append(float(element.split(",")[1].split(")")[0].strip()))


                        """Position Difference"""
                        Central_px = (max(px_list) + min(px_list)) / 2
                        Central_py = (max(py_list) + min(py_list)) / 2 
                        #m_dict[f"Port_{p_id}"] = p_list
                        #m_dict[f"Port_{p_id}_inf"] = line.split(";")[-1].strip("\n")
                        #m_dict[f"Port_{p_id}_Cnetral_x"] = Central_px
                        #m_dict[f"Port_{p_id}_Cnetral_y"] = Central_py
                        m_dict[f"Port_{p_id}_Delta_x"] = Central_px - Central_x
                        m_dict[f"Port_{p_id}_Delta_y"] = Central_py - Central_y
                        p_id += 1


                    elif line.split(":")[0] == "Module" or not line:
                        m_list.append(m_dict)
                        flag = 1
                        m_dict = {}
                        break
                
            if not line:
                break
            if flag == 1:
                continue
            line = f.readline()

    #calculate whole area of all the macros
    # for i in range(len(m_list)): 
        # whole_area+=m_list[i]['Module_Area']
    return X_factor, Y_factor, origin_point, canvas_range, m_list

"""Guangxi"""
def Coordinates_transform(result, grid_size_n):
    """
    Input:self.result   布局完成后的从macro 0到49的在32*32学习空间上的坐标
    Output:
    一个列表，列表里的每一个元素是每一个macro在实际canvas中的中心点坐标，用于输出提供给华大布线器的txt
    """

    """Guangxi"""
    X_factor, Y_factor, origin_point_canvas, canvas_range, m_list=read_file('./placement_info.txt',1, grid_size_n)
    #Valid_final_list=[]
    #Valid_macro_area=0
    #In_Canvas_range_list=[]
    Macro_center_point_list=[]
    Ports_of_macro_list=[]
    for i in range(len(result)):
        #macro_point=[]
        origin_point=[]
        #left_up_point=[]
        #right_up_point=[]
        #right_down_point=[]
        center_point=[]
        port1_point=[]
        port2_point=[]
        port3_point=[]
        for j in range(len(m_list)):    
            if(m_list[j]['Module ID']==i):
                x_origin_old=result[i][0]
                x_origin_new=origin_point_canvas[0]+X_factor*x_origin_old
                y_origin_old=result[i][1]
                y_origin_new=origin_point_canvas[1]+Y_factor*y_origin_old            
                x_center_new=x_origin_new+0.5*m_list[j]['Module_Lx']
                origin_point=[x_origin_new,y_origin_new]
                y_center_new=y_origin_new+0.5*m_list[j]['Module_Ly']
                center_point=[x_center_new,y_center_new]
                Macro_center_point_list.append(center_point)
                
                port1_point_raw_x=m_list[j]['Port_1_Delta_x']+x_center_new
                port1_point_raw_y=m_list[j]['Port_1_Delta_y']+y_center_new
                port1_point=[port1_point_raw_x, port1_point_raw_y]
                
                port2_point_raw_x=m_list[j]['Port_2_Delta_x']+x_center_new
                port2_point_raw_y=m_list[j]['Port_2_Delta_y']+y_center_new
                port2_point=[port2_point_raw_x, port2_point_raw_y]
                
                port3_point_raw_x=m_list[j]['Port_3_Delta_x']+x_center_new
                port3_point_raw_y=m_list[j]['Port_3_Delta_y']+y_center_new
                port3_point=[port3_point_raw_x, port3_point_raw_y]
                
                Ports_of_macro_list.append([port1_point,port2_point,port3_point])
                #left_up_point=[origin_point[0],origin_point[1]+m_list[j]['Module_Ly']]
                #right_up_point=[origin_point[0]+m_list[j]['Module_Lx'],origin_point[1]+m_list[j]['Module_Ly']]
                #right_down_point=[origin_point[0]+m_list[j]['Module_Lx'],origin_point[1]]
        # macro_point.append(i)       #0
        # macro_point.append(origin_point) #1
        # macro_point.append(left_up_point) #2
        # macro_point.append(right_up_point) #3
        # macro_point.append(right_down_point)#4
        
        """judge if a point is out of the canvas"""
        # if (0==((macro_point[1][0]>canvas_range[0])or(macro_point[3][0]<canvas_range[1])or(macro_point[1][1]>canvas_range[2])or(macro_point[3][1]<canvas_range[3]))):
            # In_Canvas_range_list.append(macro_point)           
        
    """judge if a macro is overlapped by others"""
    # for n in range(len(In_Canvas_range_list)):
        # overlap_flag=0
        # for u in range(len(In_Canvas_range_list)):
            # if(n!=u):
                # for t in range(1,5): #1:origin(left_down_point) 2:right_up_point 
                    # if((In_Canvas_range_list[n][t][0]>=In_Canvas_range_list[u][1][0])and(In_Canvas_range_list[n][t][0]<=In_Canvas_range_list[u][3][0])and(In_Canvas_range_list[n][t][1]>=In_Canvas_range_list[u][1][1])and(In_Canvas_range_list[n][t][1]<=In_Canvas_range_list[u][3][1])):
                        # overlap_flag=1    #a macro's point in in another macro's area
        # if(0==overlap_flag):
            # #valid_point_struct=[]
            # #valid_point_struct.append(In_Canvas_range_list[n][0]) #valid macro id
            # #valid_point_struct.append(In_Canvas_range_list[n][1]) #valid macro central point
            # Valid_final_list.append(In_Canvas_range_list[n][0]+1)
    # #calculate Valid_macro_area
    # for i in range(len(Valid_final_list)):
        # valid_id=Valid_final_list[i]-1
        # for j in range(len(m_list)):
            # if(valid_id==m_list[j]['Module ID']):
                # Valid_macro_area+=m_list[j]['Module_Area']
    # Util_area_macro=Valid_macro_area/whole_area
    # Util_macro=len(Valid_final_list)/len(result)
    # #print(len(Valid_final_list))
    # print(Valid_final_list)
    #print(Util_macro)
    #print(Util_area_macro)
    #return Valid_final_list, Util_macro, Util_area_macro, Macro_center_point_list
    return Macro_center_point_list, Ports_of_macro_list
    #print(len(In_Canvas_range_list))
    #print(In_Canvas_range_list)
    
def write_best_result(Macro_center_point_list, n):
    print(n)
    #f=open('result_1.txt','w')
    f=open('result_'+str(n)+'.txt', 'w')
    for i in range(len(Macro_center_point_list)):
        x_center=Macro_center_point_list[i][0]
        y_center=Macro_center_point_list[i][1]
        line_first='Module M'+str(i+1)+'\n'
        line_second='Orient: R0'+'\n' #temporarily ignore rotation, only R0
        line_third='Position: '+'('+str(x_center)+','+str(y_center)+')'+'\n'
        f.write(line_first)
        f.write(line_second)
        f.write(line_third)
    f.close()

"""Guangxi"""
def get_range(number,n, grid_size_n):
    D_x_y=[]     #use to store each macro dx and dy
    """Guangxi"""
    X_factor=read_file('./placement_info.txt',1, grid_size_n)[0]
    Y_factor=read_file('./placement_info.txt',1, grid_size_n)[1]
    m_list=read_file('./placement_info.txt',1, grid_size_n)[4]
    for i in range(n):
        for j in range(len(m_list)):
            if(m_list[j]['Module ID']==i):
                D_x_y.append([float(m_list[j]['Module_Lx']/X_factor),float(m_list[j]['Module_Ly']/Y_factor)])
    return D_x_y[number]
#############################Only for test#####################################
