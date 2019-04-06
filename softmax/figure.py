import matplotlib.pyplot as plt

with open('./test.txt','r') as f:
    data_list = f.readlines()
    #data_list = [i.split('\n')[0] for i in data_list]
    data_list = [i.split() for i in data_list]
    data = [(float(i[0]),float(i[1]),float(i[2]),float(i[3])) for i in data_list]

x_data1 = list(filter(lambda x:x[-1]==1.0,data))
x_data2 = list(filter(lambda x:x[-1]==2.0,data))
x_data3 = list(filter(lambda x:x[-1]==3.0,data))

x_plot_1_0 = [i[0] for i in x_data1]
x_plot_1_1 = [i[1] for i in x_data1]
x_plot_1_2 = [i[2] for i in x_data1]

x_plot_2_0 = [i[0] for i in x_data2]
x_plot_2_1 = [i[1] for i in x_data2]
x_plot_2_2 = [i[2] for i in x_data2]

x_plot_3_0 = [i[0] for i in x_data3]
x_plot_3_1 = [i[1] for i in x_data3]
x_plot_3_2 = [i[2] for i in x_data3]

plt.plot(x_plot_1_1,x_plot_1_2,'ro',label='1.0')
plt.plot(x_plot_2_1,x_plot_2_2,'bo',label='2.0')
plt.plot(x_plot_3_1,x_plot_3_2,'ko',label='3.0')
plt.legend(loc='best')
plt.show()



