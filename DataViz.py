import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools


class DataViz:
    """
        A tiny class to visualize data
    """
    def __init__(self, data):
        self.data = data

    def plot(self, x, y, title, xlabel, ylabel, legend):
        plt.plot(x, y, 'o')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(legend)
        plt.show()

    def scatter(self, x, y, title, xlabel, ylabel, legend):
        plt.scatter(x, y)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(legend)
        plt.show()

    def hist(self, x, title, xlabel, ylabel):
        plt.hist(x)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def box(self, x, title, xlabel, ylabel):
        plt.boxplot(x)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def violin(self, x, title, xlabel, ylabel):
        """
        Plots a violin chart
        Args:
            x ([type]): [description]
            title ([type]): [description]
            xlabel ([type]): [description]
            ylabel ([type]): [description]
        """
        plt.violinplot(x)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def bar(self, x, title, xlabel, ylabel):
        """[Plots a bar chart]
        Args:
            x ([type]): [description]
            title ([type]): [description]
            xlabel ([type]): [description]
            ylabel ([type]): [description]
        """
        plt.bar(x)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def pie(self, x, title, xlabel, ylabel):
        """[Plots a pie chart]
        Args:
            x ([type]): [description]
            title ([type]): [description]
            xlabel ([type]): [description]
            ylabel ([type]): [description]
        """
        plt.pie(x)
        plt.title(title)
        
    def plot_decision_boundary(self):
        """Plots a decision boundary
        """
        pass
    
    def plot_history(self, history):
        """
        Returns a plots of model learning history object

        Args:
            history: Model's learning history object
        
        Returns:
            A plot of of model learning history
        """

        train_loss = history.history['loss']
        val_loss = history.history['val_loss']

        train_accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']

        # Plot the histories
        plt.plot(train_loss, label="Train Loss")
        plt.plot(val_loss, label="Val Loss")
        plt.xlabel('Epochs')
        plt.legend()

        # Plot the accuracy
        plt.figure()
        plt.plot(train_accuracy, label="Train Accuracy")
        plt.plot(val_accuracy, label="Val Accuracy")
        plt.xlabel('Epochs')
        plt.legend()
    

    def plot_lr_vs_loss(self,learning_rate, history):
        """
        Plot a semiglogx chart showing the model's learning rate vs. loss performance
        Args:
            learning_rate: the learning rate callback
            history: model training history example(history = model.fit(X,y, epochs=100))
        """
        plt.figure(figsize=self.figsize)
        plt.semilogx(learning_rate, history.history['loss'])
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Vs. Loss')
        plt.show()


    def plot_conf_matrix(conf_matrix, classes=None, figsize=(10,7), text_size=10):
            """Plots a confusion matrix chart

            Args:
                conf_matrix (array): confusion matrix
            """
            fig, ax = plt.subplots(figsize=figsize)
            
            # Normalize our confusion matrix
            conf_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=-1)[:np.newaxis]
            n_classes = conf_norm.shape[0]
            
            # Create matrix plot
            matshow = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
            fig.colorbar(matshow)
            
            # Check classes
            if classes:
                labels = classes
            else:
                labels = np.arange(conf_matrix.shape[0])
                
            # Label the classes
            ax.set(title='Confusion Matrix',
                xlabel='Predicted Label',
                ylabel='True Label',
                xticks=labels,
                yticks=labels)
            
            # Set x-axis labels to bottom
            ax.xaxis.set_label_position('bottom')
            ax.xaxis.tick_bottom()
            
            #Adjust label size
            ax.yaxis.label.set_size(text_size)
            ax.yaxis.label.set_size(text_size)
            ax.title.set_size(text_size)
            
            # Set the threshold for different colors
            threshold = (conf_matrix.max() + conf_matrix.min()) / 2
            
            # Plot text on each cell
            for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
                plt.text(j, i, f"{conf_matrix[i, j]} ({conf_norm[i,j]*100:.1f}%)", 
                        horizontalalignment="center",
                        color="white" if conf_matrix[i,j] > threshold else "black", size=15)
        