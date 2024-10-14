from gurobipy import GRB,Model,quicksum
import pandas as pd
import time

with open('instance_2.txt', 'r') as file:
    file.readline()
    second_line = file.readline().strip()
    for _ in range(4):
        file.readline()
    data = file.read()
lines = [line.strip() for line in data.split('\n') if line.strip()]
data = [list(map(int, line.split())) for line in lines]
df = pd.DataFrame(data, columns=['Customer No', 'X Coordinate', 'Y Coordinate', 'Demand', 'Ready Time', 'Due Date', 'Service Time'])
df = pd.concat([df, df.head(1)], ignore_index=True)

#Sub tour elimination must be done
def Output(m):  
    status_code = {1:'LOADED', 2:'OPTIMAL', 3:'INFEASIBLE', 4:'INF_OR_UNBD', 5:'UNBOUNDED'} 
    status = m.status
    
    print('The optimization status is ' + status_code[status])
    if status == 2:    
        print('Optimal solution:')
        for v in m.getVars():
            if v.x > 0:
                print(str(v.varName) + " = " + str(v.x))    
        print('Optimal objective value: ' + str(m.objVal) + "\n")

number_of_customers = len(df['Customer No'])
max_number_of_vehicles = number_of_customers - 2
euclidean_distances = []
for i in range(number_of_customers):
    euclidean_distances.append([((df['X Coordinate'].loc[i] - df['X Coordinate'].loc[j])**2 + (df['Y Coordinate'].loc[i] - df['Y Coordinate'].loc[j])**2)**0.5 for j in range(number_of_customers)])

capacity = int(second_line)
c_max = max([item for sublist in euclidean_distances for item in sublist])
fixed_vehicle_cost = c_max * (number_of_customers - 2) * 2

model = Model('VRPTW')
model.setParam('OutputFlag', True)

x_ijk = model.addVars([(i, j, k) for i in range(number_of_customers - 1) for j in range(1, number_of_customers) for k in range(1, max_number_of_vehicles+1) if i != j], vtype=GRB.BINARY, name="x")
w_ik = model.addVars([(i, k) for i in range(number_of_customers) for k in range(1, max_number_of_vehicles+1)], vtype=GRB.CONTINUOUS, name="w")
k_k = model.addVars([k for k in range(1, max_number_of_vehicles+1)], vtype=GRB.BINARY, name="k")

#Add objective function
for k in range(1, max_number_of_vehicles+1):
    k_k[k].obj = fixed_vehicle_cost
    for i in range(number_of_customers - 1):
        for j in range(1, number_of_customers):
            if i != j:
                x_ijk[i, j, k].obj = euclidean_distances[i][j]

for i in range(1, number_of_customers - 1):
    model.addConstr(quicksum(x_ijk[i, j, k] for j in range(1, number_of_customers) for k in range(1, max_number_of_vehicles+1) if i != j) == 1)
    #Her araç çeşidinden ve her nodedan bir tane çıkış olur
for k in range(1, max_number_of_vehicles+1):
    model.addConstr(quicksum(df['Demand'].loc[i] * quicksum(x_ijk[i, j, k] for j in range(1, number_of_customers) if i != j) for i in range(1, number_of_customers - 1)) <= capacity)
    #Demands must be satisfied
    model.addConstr(quicksum(x_ijk[0, j, k] for j in range(1, number_of_customers-1)) == 1)
    #Sourcedan tüm müşterilere her araç çeşidinden bir tane gidebilir
    for i in range(number_of_customers):
        model.addConstr(w_ik[i, k] <= df['Due Date'].loc[0])
        model.addConstr(w_ik[i, k] >= df['Ready Time'].loc[0])
        #model.addConstr(w_ik[number_of_customers-1, k] <= df['Due Date'].loc[number_of_customers-1])
        #model.addConstr(w_ik[number_of_customers-1, k] >= df['Ready Time'].loc[number_of_customers-1])
    #Depot departure and arrival adjustment constraints
    model.addConstr(quicksum(x_ijk[i, number_of_customers-1, k] for i in range(number_of_customers - 1)) == 1)
    #Depoya geri dönüş yapan tek araç olmalı

    model.addConstr(quicksum(x_ijk[i, j, k] for i in range(number_of_customers-1) for j in range(1, number_of_customers) if i !=j) <= number_of_customers*k_k[k])
    #Binary vehicle assignment constraint

    for p in range(1, number_of_customers - 1):
        model.addConstr(quicksum(x_ijk[i, p, k] for i in range(number_of_customers - 1) if i != p) == quicksum(x_ijk[p, j, k] for j in range(1, number_of_customers) if p != j))
        #Müşteriye giren ve çıkan araçlar eşit olmalı
    #for i in range(number_of_customers-1):                      #Check
    #    model.addConstr(df['Ready Time'].loc[i] * quicksum(x_ijk[i, j, k] for j in range(1, number_of_customers-1) if i != j) <= w_ik[i, k])
    #    model.addConstr(df['Due Date'].loc[i] * quicksum(x_ijk[i, j, k] for j in range(1, number_of_customers-1) if i != j) >= w_ik[i, k])
        #Müşteri ziyaret zaman aralığı kısıtı
for i in range(number_of_customers-1):
    for j in range(1, number_of_customers):
        if i != j:
            m_ij = max([0, df['Due Date'].loc[i] + df['Service Time'].loc[i] + euclidean_distances[i][j] - df['Ready Time'].loc[j]])
            for k in range(1, max_number_of_vehicles+1):
                model.addConstr(w_ik[i, k] + df['Service Time'].loc[i] + euclidean_distances[i][j] - w_ik[j, k] <= m_ij * (1 - x_ijk[i, j, k]))
                #Feasibility constraint

start_time = time.time()
model.optimize()
end_time = time.time()
optimization_time = end_time - start_time
Output(model)

lp_filename = 'vrptw.lp'
sol_filename = 'vrptw.sol'
model.write(lp_filename)
model.write(sol_filename)

import matplotlib.pyplot as plt
import random

def scatter_plot(x_ijk, df):
    assignment = []
    for i, j, k in x_ijk.keys():
        if x_ijk[i, j, k].x > 0.5:
            assignment.append((i, j, k))

    plt.scatter(df['X Coordinate'][1:], df['Y Coordinate'][1:], color='blue')
    plt.scatter(df['X Coordinate'][0], df['Y Coordinate'][0], color='red')
    vehicle_colors = {k: (random.random(), random.random(), random.random()) for k in range(1, len(df['Customer No']) - 1)}

    for i, j, k in assignment:
        color = vehicle_colors[k]
        plt.plot([df['X Coordinate'].iloc[i], df['X Coordinate'].iloc[j]],
                [df['Y Coordinate'].iloc[i], df['Y Coordinate'].iloc[j]], color=color)

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Vehicle Routing Solution')
    plt.grid(True)
    plt.show()

scatter_plot(x_ijk, df)