import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from itertools import cycle

# no_of_dataset = 13


def plot_results():  # For classification
    eval1 = np.load('Eval_all_1.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'FNR', 'Specificity', 'FPR', 'Precision', 'FDR', 'NPV', 'FOR', 'F1-score',
             'MCC']
    Graph_Terms = [0, 1, 5, 6, 7, 8, 9]
    Batch_Size = ['16', '32', '64', '128', '256']

    bar_width = 0.15
    colors = ['#219ebc', '#f77f00', '#8ac926']
    Classifier = ['DNN', 'LSTM', 'RNN', 'SC-DSVM', 'RRWO-SC-ADSVM']

    # Function to add labels with a line for a specific bar
    def add_label_with_line(x_position, height, label, color, horizontal_offset=0.5, vertical_offset=5):
        # Adding the label with horizontal and vertical line shapes
        ax.annotate(
            label,
            xy=(x_position, height), xycoords='data',  # The bar's top
            xytext=(x_position + horizontal_offset, height + vertical_offset), textcoords='data',  # Label position
            arrowprops=dict(
                arrowstyle='-', color=color,
                connectionstyle=f"angle,angleA=0,angleB=90,rad=0",
                lw=2,  # Vertical line weight
                linestyle='-',  # Style of the line
                shrinkA=0, shrinkB=0
            ),
            ha='center', va='bottom', fontsize=10, color=color
        )

    for i in range(1):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval1.shape[1:3])
            for k in range(eval1.shape[1]):
                for l in range(eval1.shape[2]):
                    if j == 9:
                        Graph[k, l] = eval1[i, k, l, Graph_Terms[j] + 4]
                    else:
                        Graph[k, l] = eval1[i, k, l, Graph_Terms[j] + 4]

            plt.plot(Batch_Size, Graph[:, 0], color='#8ac926', linewidth=5, marker='*', markerfacecolor='y',
                     markersize=12,
                     label="AOA-SC-ADSVM")
            plt.plot(Batch_Size, Graph[:, 1], color=[0.5, 0.9, 0.9], linewidth=5, marker='^', markerfacecolor=[0.5, 0.9, 0.9],
                     markersize=12,
                     label="CWO-SC-ADSVM")
            plt.plot(Batch_Size, Graph[:, 2], color='m', linewidth=5, marker='D', markerfacecolor='b',
                     markersize=12,
                     label="TOT-SC-ADSVM")
            plt.plot(Batch_Size, Graph[:, 3], color=[0.7, 0.7, 0.9], linewidth=5, marker='>', markerfacecolor=[0.7, 0.7, 0.9],
                     markersize=12,
                     label="WOA-SC-ADSVM")
            plt.plot(Batch_Size, Graph[:,4], color='k', linewidth=5, marker='<', markerfacecolor='k',
                     markersize=12,
                     label="RRWO-SC-ADSVM")
            plt.xticks(Batch_Size, ('16', '32', '64', '128', '256'), fontsize=14, fontweight='bold')
            plt.yticks(fontsize=14, fontweight='bold')
            plt.legend(loc='best', fontsize=14, prop={'weight': 'bold'})
            plt.xlabel('Batch Size', fontsize=14, fontweight='bold')
            plt.ylabel(Terms[Graph_Terms[j]], fontsize=14, fontweight='bold')
            path1 = "./Results/Authentication-%s_line.png" % (Terms[Graph_Terms[j]])
            plt.savefig(path1)
            plt.show()

            # Create the figure and axes for plotting
            fig = plt.figure()
            ax = fig.add_axes([0.11, 0.11, 0.7, 0.7])
            fig.canvas.manager.set_window_title('Method Comparison of Dataset')

            # X positions for the bars
            X = np.arange(5)

            # Create bars for each classifier
            bars1 = plt.bar(X + 0.00, Graph[:, 5], color='#219ebc', width=0.15, label=Classifier[0])
            bars2 = plt.bar(X + 0.15, Graph[:, 6], color='#f77f00', width=0.15, label=Classifier[1])
            bars3 = plt.bar(X + 0.30, Graph[:, 7], color='#8ac926', width=0.15, label=Classifier[2])
            bars4 = plt.bar(X + 0.45, Graph[:, 8], color='b', width=0.15, label=Classifier[3])
            bars5 = plt.bar(X + 0.60, Graph[:, 4], color='y', width=0.15, label=Classifier[4])
            # Remove axes outline
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(True)
            # Customizations for the plot
            plt.xticks(X + 0.25, ['16', '32', '64', '128', '256'], fontname="Arial", fontsize=14,
                       fontweight='bold', color='k')
            plt.xlabel('Batch Size', fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')

            # Add labels with lines to specific bars (example, you can adjust the bar and label positions as needed)
            add_label_with_line(X[0] + 0.00, Graph[0, 5], 'DNN', '#219ebc', horizontal_offset=0.5,
                                vertical_offset=Graph[1, 5] / 5)  # Graph[1, 8]/ 5
            add_label_with_line(X[1] + 0.15, Graph[1, 6], 'LSTM', '#f77f00', horizontal_offset=0.5,
                                vertical_offset=Graph[0, 6] / 7)
            add_label_with_line(X[2] + 0.30, Graph[2, 7], 'RNN', '#8ac926', horizontal_offset=0.5,
                                vertical_offset=Graph[0, 7] / 7)  # Graph[0, 6]/7
            add_label_with_line(X[3] + 0.45, Graph[3, 8], 'SC-DSVM', 'b', horizontal_offset=0.5,
                                vertical_offset=Graph[0, 8] / 7)  # Graph[0, 6]/7
            add_label_with_line(X[4] + 0.60, Graph[4, 4], 'RRWO-SC-ADSVM', 'y', horizontal_offset=0.5,
                                vertical_offset=Graph[0, 4] / 7)  # Graph[0, 6]/7
            plt.grid(axis='y')
            plt.yticks(fontsize=14, fontweight='bold')
            # plt.legend(fontsize=14, prop={'weight': 'bold'})

            # Show the plot
            plt.tight_layout()
            path = "./Results/Authentication-%s_bar.png" % (Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()

def plot_Error_results_3():  # Irrigation Prediciton
    eval1 = np.load('Eval_all_2.npy', allow_pickle=True)
    Terms = ['MEP', 'SMAPE', 'MASE', 'MAE', 'RMSE', 'ONENORM', 'TWONORM', 'INFINITYNORM']
    Graph_Terms = [0, 1, 2, 3, 4]
    Steps_Per_Epoch = ['100', '200', '300', '400', '500']
    for i in range(1):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros((eval1.shape[1], eval1.shape[2]))
            for k in range(eval1.shape[1]):
                for l in range(eval1.shape[2]):
                    if j == 9:
                        Graph[k, l] = eval1[i, k, l, Graph_Terms[j]]
                    else:
                        Graph[k, l] = eval1[i, k, l, Graph_Terms[j]]

            plt.plot(Steps_Per_Epoch, Graph[:, 0], color=[0.9, 0.9, 0.3], linewidth=5, marker='*', markerfacecolor='y',
                     markersize=12,
                     label="AOA-AMViT-ENet")
            plt.plot(Steps_Per_Epoch, Graph[:, 1], color=[0.5, 0.9, 0.9], linewidth=5, marker='^', markerfacecolor=[0.5, 0.9, 0.9],
                     markersize=12,
                     label="CWO-AMViT-ENet")
            plt.plot(Steps_Per_Epoch, Graph[:, 2], color='b', linewidth=5, marker='>', markerfacecolor='b',
                     markersize=12,
                     label="TOT-AMViT-ENet")
            plt.plot(Steps_Per_Epoch, Graph[:, 3], color=[0.7, 0.7, 0.9], linewidth=5, marker='<', markerfacecolor=[0.7, 0.7, 0.9],
                     markersize=12,
                     label="WOA-AMViT-ENet")
            plt.plot(Steps_Per_Epoch, Graph[:, 4], color='k', linewidth=5, marker='o', markerfacecolor='k',
                     markersize=12,
                     label="RRWO-AMViT-ENet")
            plt.xticks(Steps_Per_Epoch, ('100', '200', '300', '400', '500'), fontsize=14, fontweight='bold')
            plt.xlabel('Steps Per Epochs', fontsize=14, fontweight='bold')
            plt.ylabel(Terms[Graph_Terms[j]], fontsize=14, fontweight='bold')
            plt.legend(loc='best', fontsize=14, prop={'weight': 'bold'})
            path1 = "./Results/Irrigation_%s_line.png" % (Terms[Graph_Terms[j]])
            plt.savefig(path1)
            plt.show()


            fig = plt.figure()
            ax = fig.add_axes([0.12, 0.1, 0.8, 0.8])
            # fig.canvas.manager.set_window_title('Epochs')
            X = np.arange(5)
            ax.bar(X + 0.00, Graph[:, 5], color='fuchsia', edgecolor='k', width=0.10, label="YOLOv3")
            ax.bar(X + 0.10, Graph[:, 6], color='yellow', edgecolor='k', width=0.10, label="CNN")
            ax.bar(X + 0.20, Graph[:, 7], color='navy', edgecolor='k', width=0.10, label="YOLOv5")
            ax.bar(X + 0.30, Graph[:, 8], color='coral', edgecolor='k', width=0.10, label="MViT-ENet")
            ax.bar(X + 0.40, Graph[:, 9], color='k', edgecolor='k', width=0.10, label="RRWO-AMViT-ENet")
            plt.xticks(X + 0.10, ('100', '200', '300', '400', '500'), fontsize=14, fontweight='bold')
            plt.xlabel('Steps Per Epochs', fontsize=14, fontweight='bold')
            plt.ylabel(Terms[Graph_Terms[j]], fontsize=14, fontweight='bold')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=14, prop={'weight': 'bold'}, fancybox=True, shadow=True)
            path = "./Results/Irrigation_%s_bar_lrean.png" % (Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def plot_Error_results_4():  # Crop Yield Prediction
    Eval_all = np.load('Eval_all_4.npy', allow_pickle=True)  # Paper
    Terms = ['MEP', 'SMAPE', 'RMSE', 'MAE', 'MASE', 'ONE-NORM', 'TWO-NORM', 'INFINITY-NORM']
    Graph_Terms = [4, 5, 6]

    for u in range(1):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros((Eval_all.shape[1], Eval_all.shape[2] + 1))
            for k in range(Eval_all.shape[1]):
                for l in range(Eval_all.shape[2]):
                    if j == 9:
                        Graph[k, l] = Eval_all[u, k, l, Graph_Terms[j]]
                    else:
                        Graph[k, l] = Eval_all[u, k, l, Graph_Terms[j]]

            Act = ['Linear', 'ReLU', 'Leaky ReLU', 'TanH', 'Sigmoid', 'Softmax']
            n_groups = 6
            index = np.arange(n_groups)
            bar_width = 0.10
            opacity = 1
            fig = plt.figure()
            fig.canvas.manager.set_window_title('Activation Function')
            plt.bar(index, Graph[:, 0], bar_width, alpha=opacity, color='m', label='DNN')
            plt.bar(index + bar_width, Graph[:, 1], bar_width, alpha=opacity, color='c', label='1DCNN')
            plt.bar(index + bar_width + bar_width, Graph[:, 2], bar_width, alpha=opacity, color='hotpink',
                    label='Autoencoder')
            plt.bar(index + 3 * bar_width, Graph[:, 3], bar_width, alpha=opacity, color='y', label='LSTM')
            plt.bar(index + 4 * bar_width, Graph[:, 4], bar_width, alpha=opacity, color='k',
                    label='CAE-LSTM')
            plt.xticks(index + 0.25, ('Linear', 'ReLU', 'Leaky ReLU', 'TanH', 'Sigmoid', 'Softmax'))
            plt.ylabel(Terms[j], size=16)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path = "./Results/Crop_Yiel_Prediction_perfcls_%s.png" % (Terms[j])
            plt.savefig(path)
            plt.show()


def Table():
    from prettytable import PrettyTable
    import numpy as np

    # def Table():
    eval = np.load('Eval_all_1.npy', allow_pickle=True)
    Algorithm = ['TERMS/Kfold', 'GaOA', 'GOA', 'NBOA', 'PCOA', 'Proposed']
    Classifier = ['TERMS/Kfold', 'CNN', 'RESNET', 'DENSENET', 'RES-DENSENET', 'VIT-DRDNet']
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Table_Terms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    table_terms = [Terms[i] for i in Table_Terms]
    Epoch = [10, 20, 30, 40, 50]
    Epoch = [1, 2, 3, 4, 5 ]
    # for i in range(eval.shape[0]):
    for k in range(len(Table_Terms)):
        value = eval[0, :, :, 4:]

        # Table = PrettyTable()
        # Table.add_column(Algorithm[0], Epoch)
        # for j in range(len(Algorithm) - 1):
        #     Table.add_column(Algorithm[j + 1], value[:, j, k])
        # print('------------------------------- Dataset- ', i + 1, table_terms[k], '  Algorithm Comparison',
        #       '---------------------------------------')
        # print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Epoch)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1, k])
        print('------------------------------- Dataset- ', table_terms[k], '  Classifier Comparison',
              '---------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Epoch)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[:, j, k])
        print('------------------------------- Dataset', table_terms[k], '  Algorithm Comparison',
              '---------------------------------------')
        print(Table)




def plot_Table():  # Irrigation Prediciton Table
    eval1 = np.load('Eval_learning.npy', allow_pickle=True)
    Terms = ['MEP', 'SMAPE', 'MASE', 'MAE', 'RMSE', 'ONENORM', 'TWONORM', 'INFINITYNORM']
    Classifier = ['TERMS', 'DNN', '1DCNN', 'Autoencoder', 'LSTM', 'CAE-LSTM']

    for i in range(1):
        value1 = eval1[i, 4, :, :]

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value1[j, :])
        print('--------------------------------------------------Learning Percentage- Irrigation pridiction '
              'Classifier Comparison',
              '--------------------------------------------------')
        print(Table)


def Plot_ROC_Curve():  # Classification
    lw = 2
    cls = ['VGG16', 'ShuffleNet', 'Autoencoder', 'LSTM', 'CAE-LSTM']
    Actual = np.load('Target_1.npy', allow_pickle=True)
    fig = plt.figure()
    fig.canvas.manager.set_window_title('ROC Curve')
    colors = cycle(["blue", "darkorange", "limegreen", "deeppink", "black"])
    for i, color in zip(range(5), colors):  # For all classifiers
        Predicted = np.load('Y_Score.npy', allow_pickle=True)[0][i]
        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
        plt.plot(
            false_positive_rate1,
            true_positive_rate1,
            color=color,
            lw=lw,
            label=cls[i])

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Accuracy')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    path = "./Results/Crop_Pred_ROC.png"
    plt.savefig(path)
    plt.show()


def Plot_irrigation_ROC_Curve():
    lw = 2
    cls = ['DNN', '1DCNN', 'Autoencoder', 'LSTM', 'CAE-LSTM']
    Actual = np.load('Targets_3.npy', allow_pickle=True)
    fig = plt.figure()
    fig.canvas.manager.set_window_title('ROC Curve')
    colors = cycle(["green", "orange", "deeppink", "y", "black"])
    for i, color in zip(range(5), colors):  # For all classifiers
        Predicted = np.load('Y_Scores.npy', allow_pickle=True)[0][i]
        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
        plt.plot(
            false_positive_rate1,
            true_positive_rate1,
            color=color,
            lw=lw,
            label=cls[i])

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Accuracy')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    path = "./Results/Irration_ROC.png"
    plt.savefig(path)
    plt.show()


def plot_Epoch_Results():  # For classification
    eval1 = np.load('Eval_all_Epoch.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'FNR', 'Specificity', 'FPR', 'Precision', 'FDR', 'NPV', 'FOR', 'F1_score',
             'MCC']
    Graph_Terms = [0, 1, 4, 6, 8]
    for i in range(1):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval1.shape[1:3])
            for k in range(eval1.shape[1]):
                for l in range(eval1.shape[2]):
                    if j == 9:
                        Graph[k, l] = eval1[i, k, l, Graph_Terms[j] + 4]
                    else:
                        Graph[k, l] = eval1[i, k, l, Graph_Terms[j] + 4]

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            fig.canvas.manager.set_window_title('Epochs')
            X = np.arange(5)

            ax.bar(X + 0.00, Graph[:, 0], color='#FFC125', edgecolor='k', width=0.10, label="VGG16")
            ax.bar(X + 0.10, Graph[:, 1], color='#8DEEEE', edgecolor='k', width=0.10, label="ShuffleNet")
            ax.bar(X + 0.20, Graph[:, 2], color='#FF1493', edgecolor='k', width=0.10, label="Autoencoder")
            ax.bar(X + 0.30, Graph[:, 3], color='lime', edgecolor='k', width=0.10, label="LSTM")
            ax.bar(X + 0.40, Graph[:, 4], color='k', edgecolor='k', width=0.10, label="CAE-LSTM")
            plt.xticks(X + 0.10, ('100', '200', '300', '400', '500'))
            plt.xlabel('Epochs')
            plt.ylabel(Terms[Graph_Terms[j]])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path = "./Results/Crop_Prediction_%s_Epoch_bar_net.png" % (Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()



def plot_Results_Encryption():
    for a in range(1):
        Eval =np.load('Evaluate_all.npy',allow_pickle=True)[a]

        Classifier = ['DNN', 'LSTM', 'RNN', 'SC-DSVM', 'RRWO-SC-ADSVM']
        Terms = ['Transaction Throughput (%)','Latency (Transaction Finalization Time) (s)','Security (%)', 'Smart Contract Capabilities', 'Energy Efficiency (J)']
        for b in range(len(Terms)):
            # learnper = [1, 2, 3, 4, 5]
            #
            # X = np.arange(5)
            # plt.plot(learnper, Eval[:, 0,b], color='#aaff32', linewidth=3, marker='o', markerfacecolor='#aaff32', markersize=14,
            #          label="AOA-SC-ADSVM")
            # plt.plot(learnper, Eval[:, 1,b], color='#ad03de', linewidth=3, marker='o', markerfacecolor='#ad03de', markersize=14,
            #          label="CWO-SC-ADSVM")
            # plt.plot(learnper, Eval[:, 2,b], color='#8c564b', linewidth=3, marker='o', markerfacecolor='#8c564b', markersize=14,
            #          label="TOT-SC-ADSVM")
            # plt.plot(learnper, Eval[:, 3,b], color='#ff000d', linewidth=3, marker='o', markerfacecolor='#ff000d', markersize=14,
            #          label="WOA-SC-ADSVM")
            # plt.plot(learnper, Eval[:, 4,b], color='k', linewidth=3, marker='o', markerfacecolor='k', markersize=14,
            #          label="RRWO-SC-ADSVM")
            #
            # labels = ['5', '10', '15', '20', '25']
            # plt.xticks(learnper, labels)
            #
            # plt.xlabel('BLOCK SIZE')
            # plt.ylabel(Terms[b])
            # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
            #            ncol=3, fancybox=True, shadow=True)
            # path1 = "./Results/bloch chain-%s-line.png" % (Terms[b])
            # plt.savefig(path1)
            # plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.13, 0.13, 0.75, 0.75])
            X = np.arange(5)
            ax.bar(X + 0.00, Eval[:, 5,b], color='#aaff32', width=0.10, label="Bitcoin")
            ax.bar(X + 0.10, Eval[:, 6,b], color='#ad03de', width=0.10, label="Hyperledger Fabric")
            ax.bar(X + 0.20, Eval[:, 7,b], color='#8c564b', width=0.10, label="Solana")
            ax.bar(X + 0.30, Eval[:, 8,b], color='#ff000d', width=0.10, label="Polkadot")
            ax.bar(X + 0.40, Eval[:, 9,b], color='k', width=0.10, label="Ethereum")
            # plt.xticks(X + 0.25, ('5', '10', '15', '20', '25'))


            labels = ['1', '10', '100', '1000', '10000']
            # labels = ['12', '0.5', '600', '2.5', '0.2']
            plt.xticks(X+0.20, labels, fontsize=14, fontweight='bold')


            # plt.xlabel('BLOCK SIZE')
            plt.ylabel(Terms[b], fontsize=12, fontweight='bold')
            # plt.xscale("log")
            # plt.yscale("log")
            plt.xlabel("Block Time (seconds)", fontsize=14, fontweight='bold')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, prop={'weight': 'bold'}, fontsize=14, fancybox=True, shadow=True)
            path1 = "./Results/bloch chain-%s-bar.png" % (Terms[b])
            plt.savefig(path1)
            plt.show()

def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def plot_results_conv():
    Result = np.load('Fitness.npy', allow_pickle=True)
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'AOA-SC-ADSVM', 'CWO-SC-ADSVM', 'TOT-SC-ADSVM', 'WOA-SC-ADSVM', 'RRWO-SC-ADSVM']

    for i in range(Result.shape[0]):
        Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):  # for 5 algms
            Conv_Graph[j, :] = stats(Fitness[i, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Dataset', i + 1, 'Statistical Report ',
              '--------------------------------------------------')

        print(Table)

        length = np.arange(50)
        Conv_Graph = Fitness[i]

        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red', markersize=12,
                 label=Algorithm[1])
        plt.plot(length, Conv_Graph[1, :], color='c', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12,
                 label=Algorithm[2])
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='cyan',
                 markersize=12,
                 label=Algorithm[3])
        plt.plot(length, Conv_Graph[3, :], color='y', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12,
                 label=Algorithm[4])
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12,
                 label=Algorithm[5])
        plt.ylabel('Cost Function')
        plt.xlabel('no of Iteration')
        plt.legend(loc=1)
        plt.savefig("Dataset_%s_conv.png" % (i + 1))
        plt.show()



if __name__ == '__main__':
    # plot_results()
    # plot_Error_results_3()
    # plot_Results_Encryption()
    plot_results_conv()
