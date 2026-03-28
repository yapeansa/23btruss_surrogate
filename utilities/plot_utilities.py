import numpy as np
import matplotlib.pyplot as plt

def plot_data_general(np_array_1=None, np_array_2=None, np_array_3=None, savefile_name=None, labels=None, axis_label=None):
    plt.figure(figsize=(12, 4))
    x_axis = np_array_1
    fem_data = np_array_2
    femnn_data = np_array_3
    
    y_min  = np.amin(fem_data)
    y_max  = np.amax(fem_data)

    print(f'O mínimo é: {y_min}')
    print(f'O máximo é: {y_max}')

    # plot force coefficients 
    plt.plot(x_axis, fem_data, label= labels[0], color='darkmagenta', linestyle='solid', markeredgecolor='darkmagenta', markerfacecolor='darkmagenta')    
    plt.plot(x_axis, femnn_data, label=labels[1], color='olive', linestyle='dashed', marker='X', markeredgecolor='olive',
        markerfacecolor='olive')

    # plt.yticks(np.arange(np.ceil(-80/20)*20, y_max+1, 20))
    plt.xlabel(axis_label[0])
    plt.ylabel(axis_label[1])
    plt.grid(True, c='#ccc', alpha=0.5)
    plt.legend(loc='upper center')

    # save figure 
    plt.savefig(f'{savefile_name}.pdf', dpi=300, bbox_inches='tight', format='pdf')
    plt.show()
    plt.close()
